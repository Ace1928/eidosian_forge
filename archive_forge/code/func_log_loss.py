from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['losses.log_loss'])
@dispatch.add_dispatch_support
def log_loss(labels, predictions, weights=1.0, epsilon=1e-07, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Adds a Log Loss term to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    epsilon: A small increment to add to avoid taking a log of zero.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'log_loss', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = -math_ops.multiply(labels, math_ops.log(predictions + epsilon)) - math_ops.multiply(1 - labels, math_ops.log(1 - predictions + epsilon))
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)