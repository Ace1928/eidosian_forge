import functools
import weakref
from enum import Enum
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import losses_utils
from keras.src.utils import tf_utils
from keras.src.utils.generic_utils import to_list
def binary_matches(y_true, y_pred, threshold=0.5):
    """Creates int Tensor, 1 for label-prediction match, 0 for mismatch.

    Args:
      y_true: Ground truth values, of shape (batch_size, d0, .. dN).
      y_pred: The predicted values, of shape (batch_size, d0, .. dN).
      threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.

    Returns:
      Binary matches, of shape (batch_size, d0, .. dN).
    """
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    return tf.cast(tf.equal(y_true, y_pred), backend.floatx())