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
def test_multi_dim_weights(self):
    """Asserts evaluation metrics for multi-dimensional input and logits."""
    global_step = 100
    create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), global_step, self._model_dir)
    label_dimension = 3
    dnn_regressor = self._dnn_regressor_fn(hidden_units=(2, 2), feature_columns=[self._fc_impl.numeric_column('age', shape=[2])], label_dimension=label_dimension, weight_column='w', model_dir=self._model_dir)

    def _input_fn():
        return ({'age': [[10.0, 8.0]], 'w': [10.0]}, [[1.0, -1.0, 0.5]])
    expected_loss = 43.929
    metrics = dnn_regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertAlmostEqual(expected_loss, metrics[metric_keys.MetricKeys.LOSS], places=3)