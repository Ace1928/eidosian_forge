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
def get_shape_list(tensor, expected_rank=None, name=None):
    """official.modeling.tf_utils.get_shape_list"""
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape