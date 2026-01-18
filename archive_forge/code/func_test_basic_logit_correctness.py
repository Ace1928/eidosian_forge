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
def test_basic_logit_correctness(self):
    """linear_logit_fn simply wraps feature_column_lib.linear_model."""
    age = self._fc_lib.numeric_column('age')
    with tf.Graph().as_default():
        logit_fn = linear.linear_logit_fn_builder(units=2, feature_columns=[age])
        logits = logit_fn(features={'age': [[23.0], [31.0]]})
        bias_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, 'linear_model/bias_weights')[0]
        age_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, 'linear_model/age')[0]
        with tf.compat.v1.Session() as sess:
            sess.run([tf.compat.v1.initializers.global_variables()])
            self.assertAllClose([[0.0, 0.0], [0.0, 0.0]], logits.eval())
            sess.run(bias_var.assign([10.0, 5.0]))
            self.assertAllClose([[10.0, 5.0], [10.0, 5.0]], logits.eval())
            sess.run(age_var.assign([[2.0, 3.0]]))
            self.assertAllClose([[56.0, 74.0], [72.0, 98.0]], logits.eval())