from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def assert_close(expected, actual, rtol=0.0001, name='assert_close'):
    with ops.name_scope(name, 'assert_close', (expected, actual, rtol)) as scope:
        expected = ops.convert_to_tensor(expected, name='expected')
        actual = ops.convert_to_tensor(actual, name='actual')
        rdiff = tf.math.abs(expected - actual, 'diff') / tf.math.abs(expected)
        rtol = ops.convert_to_tensor(rtol, name='rtol')
        return tf.compat.v1.debugging.assert_less(rdiff, rtol, data=('Condition expected =~ actual did not hold element-wise:expected = ', expected, 'actual = ', actual, 'rdiff = ', rdiff, 'rtol = ', rtol), name=scope)