import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def get_text_data():
    train = np.array([['This is a test example'], ['This is another text example'], ['Is this another example?'], [''], ['Is this a long long long long long long example?']], dtype=np.str)
    test = np.array([['This is a test example'], ['This is another text example'], ['Is this another example?']], dtype=np.str)
    y = np.random.rand(3, 1)
    return (train, test, y)