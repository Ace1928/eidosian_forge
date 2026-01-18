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
def test_features_tensor_raises_value_error(self):
    """Tests that passing a Tensor for features raises a ValueError."""
    hidden_units = (2, 2)
    logits_dimension = 3
    inputs = ([[10.0]], [[8.0]])
    expected_logits = [[0, 0, 0]]
    with tf.Graph().as_default():
        tf.compat.v1.train.create_global_step()
        head = mock_head(self, hidden_units=hidden_units, logits_dimension=logits_dimension, expected_logits=expected_logits)
        with self.assertRaisesRegexp(ValueError, 'features should be a dict'):
            self._dnn_model_fn(features=tf.constant(inputs), labels=tf.constant([[1]]), mode=ModeKeys.TRAIN, head=head, hidden_units=hidden_units, feature_columns=[self._fc_impl.numeric_column('age', shape=np.array(inputs).shape[1:])], optimizer=mock_optimizer(self, hidden_units))