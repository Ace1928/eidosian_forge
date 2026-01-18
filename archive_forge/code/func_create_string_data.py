import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
def create_string_data(length, num_entries, vocabulary, pct_oov, oov_string='__OOV__'):
    """Create a ragged tensor with random data entries."""
    lengths = (np.random.random(size=num_entries) * length).astype(int)
    total_length = np.sum(lengths)
    num_oovs = int(pct_oov * total_length)
    values = []
    for _ in range(total_length):
        values.append(random.choice(vocabulary))
    if pct_oov > 0:
        oov_cadence = int(total_length / num_oovs)
        idx = 0
        for _ in range(num_oovs):
            if idx < total_length:
                values[idx] = oov_string
            idx += oov_cadence
    return tf.RaggedTensor.from_row_lengths(values, lengths)