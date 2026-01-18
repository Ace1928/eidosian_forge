import numpy as np
import pytest
import tensorflow as tf
from autokeras.utils import data_utils
def test_cast_to_string_with_float32():
    tensor = tf.constant([0.1, 0.2], dtype=tf.float32)
    assert tf.string == data_utils.cast_to_string(tensor).dtype