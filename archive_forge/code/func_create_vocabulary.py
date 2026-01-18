import itertools
import math
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
def create_vocabulary(vocab_size):
    base = len(string.ascii_letters)
    n = math.ceil(math.log(vocab_size, base))
    vocab = []
    for i in range(1, n + 1):
        for item in itertools.product(string.ascii_letters, repeat=i):
            if len(vocab) >= vocab_size:
                break
            vocab.append(''.join(item))
    return vocab