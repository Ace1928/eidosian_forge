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
def dnn_model_fn_v2(features, labels, mode, head, hidden_units, feature_columns, optimizer='Adagrad', activation_fn=tf.nn.relu, dropout=None, config=None, use_tpu=False, batch_norm=False):
    """Deep Neural Net model_fn v2.

  This function is different than _dnn_model_fn_v1 in the way it handles the
  optimizer when a String optimizer name is passed.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `base_head.Head` instance.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    optimizer: String, `tf.keras.optimizers.Optimizer` object, or callable that
      creates the optimizer to use for training. If not specified, will use the
      Adagrad optimizer. If it is String, the default learning rate of the
      optimizer will be used. If it is String, and optimizer does not have a
      default learning rate, then, a fixed learning rate of 0.05 is used.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    config: `RunConfig` object to configure the runtime settings.
    use_tpu: Whether to make a DNN model able to run on TPU. Will make function
      return a `_TPUEstimatorSpec` instance and disable variable partitioning.
    batch_norm: Whether to use batch normalization after each hidden layer.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If features has the wrong type.
  """
    _validate_features(features)
    del config
    logits, trainable_variables, update_ops = _dnn_model_fn_builder_v2(units=head.logits_dimension, hidden_units=hidden_units, feature_columns=feature_columns, activation_fn=activation_fn, dropout=dropout, batch_norm=batch_norm, features=features, mode=mode)
    if mode == ModeKeys.TRAIN:
        optimizer = optimizers.get_optimizer_instance_v2(optimizer)
        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    if use_tpu:
        estimator_spec_fn = head._create_tpu_estimator_spec
    else:
        estimator_spec_fn = head.create_estimator_spec
    return estimator_spec_fn(features=features, mode=mode, labels=labels, optimizer=optimizer, logits=logits, trainable_variables=trainable_variables, update_ops=update_ops)