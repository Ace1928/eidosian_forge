import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output