import random
import tempfile
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import string_lookup
def define_model(self):
    """A simple model for test of tf.distribute + KPL."""
    model_input = keras.layers.Input(shape=(3,), dtype=tf.int64, name='model_input')
    emb_output = keras.layers.Embedding(input_dim=len(self.FEATURE_VOCAB) + 2, output_dim=20)(model_input)
    emb_output = tf.reduce_mean(emb_output, axis=1)
    dense_output = keras.layers.Dense(units=1, activation='sigmoid')(emb_output)
    model = keras.Model({'features': model_input}, dense_output)
    return model