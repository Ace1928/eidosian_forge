from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def dnn_logit_fn(features, mode):
    """Deep Neural Network logit_fn.

    Args:
      features: This is the first item returned from the `input_fn` passed to
        `train`, `evaluate`, and `predict`. This should be a single `Tensor` or
        `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
        `ModeKeys`.

    Returns:
      A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    """
    dnn_model = _DNNModelV2(units, hidden_units, feature_columns, activation_fn, dropout, batch_norm, name='dnn')
    return dnn_model(features, mode)