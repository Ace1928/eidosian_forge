from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def create_estimator_spec_summary(regularized_training_loss, regularization_losses=None, summary_key_fn=None):
    """Create summary for estimator_spec."""
    with ops.name_scope(''):
        keys = metric_keys.MetricKeys
        loss_key = summary_key_fn(keys.LOSS) if summary_key_fn else keys.LOSS
        tf.compat.v1.summary.scalar(loss_key, regularized_training_loss)
        if regularization_losses is not None:
            regularization_loss = tf.math.add_n(regularization_losses)
            regularization_loss_key = summary_key_fn(keys.LOSS_REGULARIZATION) if summary_key_fn else keys.LOSS_REGULARIZATION
            tf.compat.v1.summary.scalar(regularization_loss_key, regularization_loss)