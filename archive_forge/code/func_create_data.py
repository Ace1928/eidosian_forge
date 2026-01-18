import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
def create_data(length, num_entries, max_value, dtype):
    """Create a ragged tensor with random data entries."""
    lengths = (np.random.random(size=num_entries) * length).astype(int)
    total_length = np.sum(lengths)
    values = (np.random.random(size=total_length) * max_value).astype(dtype)
    return tf.RaggedTensor.from_row_lengths(values, lengths)