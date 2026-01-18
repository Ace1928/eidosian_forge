from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def create_loss(self, features, mode, logits=None, labels=None):
    """See `_Head`."""
    model_outputs = self.state_manager.define_loss(self.model, features, mode)
    tf.compat.v1.summary.scalar(head_lib._summary_key(self._name, metric_keys.MetricKeys.LOSS), model_outputs.loss)
    return model_outputs