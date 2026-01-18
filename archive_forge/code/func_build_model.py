from __future__ import print_function
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from filelock import FileLock
import os
import argparse
import tarfile
import numpy as np
import re
from ray import train, tune
def build_model(self):
    """Helper method for creating the model"""
    vocab = set()
    for story, q, answer in self.train_stories + self.test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)
    vocab_size = len(vocab) + 1
    story_maxlen = max((len(x) for x, _, _ in self.train_stories + self.test_stories))
    query_maxlen = max((len(x) for _, x, _ in self.train_stories + self.test_stories))
    word_idx = {c: i + 1 for i, c in enumerate(vocab)}
    self.inputs_train, self.queries_train, self.answers_train = vectorize_stories(word_idx, story_maxlen, query_maxlen, self.train_stories)
    self.inputs_test, self.queries_test, self.answers_test = vectorize_stories(word_idx, story_maxlen, query_maxlen, self.test_stories)
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
    input_encoder_m.add(Dropout(self.config.get('dropout', 0.3)))
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
    input_encoder_c.add(Dropout(self.config.get('dropout', 0.3)))
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
    question_encoder.add(Dropout(self.config.get('dropout', 0.3)))
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)
    response = add([match, input_encoded_c])
    response = Permute((2, 1))(response)
    answer = concatenate([response, question_encoded])
    answer = LSTM(32)(answer)
    answer = Dropout(self.config.get('dropout', 0.3))(answer)
    answer = Dense(vocab_size)(answer)
    answer = Activation('softmax')(answer)
    model = Model([input_sequence, question], answer)
    return model