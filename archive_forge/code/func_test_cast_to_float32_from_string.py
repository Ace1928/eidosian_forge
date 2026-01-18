import numpy as np
import pytest
import tensorflow as tf
from autokeras.utils import data_utils
def test_cast_to_float32_from_string():
    tensor = tf.constant(['0.3'], dtype=tf.string)
    assert tf.float32 == data_utils.cast_to_float32(tensor).dtype