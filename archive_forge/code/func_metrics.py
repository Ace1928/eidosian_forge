from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def metrics(self, regularization_losses=None):
    """Creates metrics. See `base_head.Head` for details."""
    eval_metrics = {}
    keys = metric_keys.MetricKeys
    if regularization_losses is not None:
        eval_metrics[self._loss_regularization_key] = tf.keras.metrics.Mean(name=keys.LOSS_REGULARIZATION)
    with ops.name_scope('merge_eval'):
        for loss_key in self._loss_keys:
            eval_metrics[loss_key] = tf.keras.metrics.Mean(name=loss_key)
    return eval_metrics