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
def test_compute_fraction_of_zero_v2(self):
    """Tests the calculation of sparsity."""
    if self._fc_lib != feature_column_v2:
        return
    age = tf.feature_column.numeric_column('age')
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=5)
    with tf.Graph().as_default():
        model = feature_column_v2.LinearModel(feature_columns=[age, occupation], units=3, name='linear_model')
        features = {'age': [[23.0], [31.0]], 'occupation': [['doctor'], ['engineer']]}
        model(features)
        variables = model.variables
        variables.remove(model.bias)
        fraction_zero = linear._compute_fraction_of_zero(variables)
        age_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, 'linear_model/age')[0]
        with tf.compat.v1.Session() as sess:
            sess.run([tf.compat.v1.initializers.global_variables()])
            self.assertAllClose(1, fraction_zero.eval())
            sess.run(age_var.assign([[2.0, 0.0, -1.0]]))
            self.assertAllClose(16.0 / 18.0, fraction_zero.eval())