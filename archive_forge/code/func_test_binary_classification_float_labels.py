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
def test_binary_classification_float_labels(self):
    base_global_step = 100
    hidden_units = (2, 2)
    create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
    expected_loss = 1.781721
    opt = mock_optimizer(self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_classifier = self._dnn_classifier_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), optimizer=opt, model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)
    num_steps = 5
    dnn_classifier.train(input_fn=lambda: ({'age': [[10.0]]}, [[0.8]]), steps=num_steps)
    self.assertEqual(1, opt.minimize.call_count)