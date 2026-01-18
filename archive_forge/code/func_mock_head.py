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
def mock_head(testcase, hidden_units, logits_dimension, expected_logits):
    """Returns a mock head that validates logits values and variable names."""
    hidden_weights_names = [(HIDDEN_WEIGHTS_NAME_PATTERN + '/part_0:0') % i for i in range(len(hidden_units))]
    hidden_biases_names = [(HIDDEN_BIASES_NAME_PATTERN + '/part_0:0') % i for i in range(len(hidden_units))]
    expected_var_names = hidden_weights_names + hidden_biases_names + [LOGITS_WEIGHTS_NAME + '/part_0:0', LOGITS_BIASES_NAME + '/part_0:0']

    def _create_tpu_estimator_spec(features, mode, logits, labels, train_op_fn=None, optimizer=None):
        del features, labels
        trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        testcase.assertItemsEqual(expected_var_names, [var.name for var in trainable_vars])
        loss = tf.constant(1.0)
        assert_logits = assert_close(expected_logits, logits, message='Failed for mode={}. '.format(mode))
        with tf.control_dependencies([assert_logits]):
            if mode == ModeKeys.TRAIN:
                if train_op_fn is not None:
                    train_op = train_op_fn(loss)
                elif optimizer is not None:
                    train_op = optimizer.minimize(loss, global_step=None)
                return model_fn._TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            elif mode == ModeKeys.EVAL:
                return model_fn._TPUEstimatorSpec(mode=mode, loss=tf.identity(loss))
            elif mode == ModeKeys.PREDICT:
                return model_fn._TPUEstimatorSpec(mode=mode, predictions={'logits': tf.identity(logits)})
            else:
                testcase.fail('Invalid mode: {}'.format(mode))

    def _create_estimator_spec(features, mode, logits, labels, train_op_fn=None, optimizer=None):
        tpu_spec = _create_tpu_estimator_spec(features, mode, logits, labels, train_op_fn, optimizer)
        return tpu_spec.as_estimator_spec()
    head = tf.compat.v1.test.mock.NonCallableMagicMock(spec=head_lib._Head)
    head.logits_dimension = logits_dimension
    head._create_tpu_estimator_spec = tf.compat.v1.test.mock.MagicMock(wraps=_create_tpu_estimator_spec)
    head.create_estimator_spec = tf.compat.v1.test.mock.MagicMock(wraps=_create_estimator_spec)
    return head