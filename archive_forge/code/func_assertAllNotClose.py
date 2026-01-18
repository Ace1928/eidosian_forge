from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def assertAllNotClose(self, t1, t2):
    """Helper assert for arrays."""
    sum_of_abs_diff = 0.0
    for x, y in zip(t1, t2):
        try:
            for a, b in zip(x, y):
                sum_of_abs_diff += abs(b - a)
        except TypeError:
            sum_of_abs_diff += abs(y - x)
    self.assertGreater(sum_of_abs_diff, 0)