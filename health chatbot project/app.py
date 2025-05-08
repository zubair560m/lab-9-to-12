from gensim.models import Word2Vec
import streamlit as st
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load Word2Vec model
word2vec_model = Word2Vec.load("word2vec_model.bin")

# Load necessary NLTK data
nltk.download('punkt')

# Load the trained LSTM model
model = load_model("chatbot_lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load dataset for label mapping
df = pd.read_csv("C:/Users/DELL/Downloads/train_data_chatbot.csv")

# Encode labels if not already encoded
if 'label' not in df.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['short_answer'])
else:
    unique_labels = df['label'].unique()
    df = df[df['label'].isin(unique_labels)]

# Get max sequence length used during training
max_seq_length = max(len(seq) for seq in tokenizer.texts_to_sequences(df['short_question'].astype(str)))

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

# Function to predict answers
def get_response(user_input):
    tokens = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([" ".join(tokens)])
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post')

    # Get model prediction
    prediction = model.predict(padded_sequence, verbose=0)
    predicted_label = np.argmax(prediction)

    # Get answers with the predicted label
    possible_answers = df[df['label'] == predicted_label]['short_answer'].values
    if len(possible_answers) > 0:
        return np.random.choice(possible_answers)  # Random to avoid always same answer
    else:
        return "I'm not sure. Please consult a doctor."

# Streamlit UI
st.title("\U0001FA7A General Health Query Chatbot")
st.write("Ask any health-related question, and I'll try my best to provide helpful answers!")

# User input box
user_query = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_query.strip():
        response = get_response(user_query)
        st.success(f"**Chatbot:** {response}")
    else:
        st.warning("Please enter a valid health-related question.")
