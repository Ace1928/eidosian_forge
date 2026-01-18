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
def testFromCheckpointMultiBatch(self):
    age_weight = 10.0
    bias = 5.0
    initial_global_step = 100
    with tf.Graph().as_default():
        tf.Variable([[age_weight]], name=AGE_WEIGHT_NAME)
        tf.Variable([bias], name=BIAS_NAME)
        tf.Variable(initial_global_step, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    mock_optimizer = self._mock_optimizer(expected_loss=52004.0)
    linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir, optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)
    num_steps = 10
    linear_regressor.train(input_fn=lambda: ({'age': ((17,), (15,))}, ((5.0,), (3.0,))), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(expected_global_step=initial_global_step + num_steps, expected_age_weight=age_weight, expected_bias=bias)