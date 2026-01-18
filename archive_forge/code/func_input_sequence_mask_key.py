from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.head import multi_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@property
def input_sequence_mask_key(self):
    """Returns the key for the sequence mask feature."""
    return self._sequence_length_mask