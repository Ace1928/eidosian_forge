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
def mock_optimizer(testcase, hidden_units, expected_loss=None):
    """Creates a mock optimizer to test the train method.

  Args:
    testcase: A TestCase instance.
    hidden_units: Iterable of integer sizes for the hidden layers.
    expected_loss: If given, will assert the loss value.

  Returns:
    A mock Optimizer.
  """
    hidden_weights_names = [(HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i for i in range(len(hidden_units))]
    hidden_biases_names = [(HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i for i in range(len(hidden_units))]
    expected_var_names = hidden_weights_names + hidden_biases_names + [LOGITS_WEIGHTS_NAME + '/part_0:0', LOGITS_BIASES_NAME + '/part_0:0']

    def _minimize(loss, global_step=None, var_list=None):
        """Mock of optimizer.minimize."""
        trainable_vars = var_list or tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        testcase.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
        testcase.assertEquals(0, loss.shape.ndims)
        if expected_loss is None:
            if global_step is not None:
                return tf.compat.v1.assign_add(global_step, 1).op
            return tf.no_op()
        assert_loss = assert_close(tf.cast(expected_loss, name='expected', dtype=tf.dtypes.float32), loss, name='assert_loss')
        with tf.control_dependencies((assert_loss,)):
            if global_step is not None:
                return tf.compat.v1.assign_add(global_step, 1).op
            return tf.no_op()
    optimizer_mock = tf.compat.v1.test.mock.NonCallableMagicMock(spec=tf.compat.v1.train.Optimizer, wraps=tf.compat.v1.train.Optimizer(use_locking=False, name='my_optimizer'))
    optimizer_mock.minimize = tf.compat.v1.test.mock.MagicMock(wraps=_minimize)
    return optimizer_mock