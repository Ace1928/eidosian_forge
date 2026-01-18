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
@tf_export(v1=['losses.cosine_distance'])
@dispatch.add_dispatch_support
@deprecated_args(None, 'dim is deprecated, use axis instead', 'dim')
def cosine_distance(labels, predictions, axis=None, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS, dim=None):
    """Adds a cosine-distance loss to the training procedure.

  Note that the function assumes that `predictions` and `labels` are already
  unit-normalized.

  Args:
    labels: `Tensor` whose shape matches 'predictions'
    predictions: An arbitrary matrix.
    axis: The dimension along which the cosine distance is computed.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: Type of reduction to apply to loss.
    dim: The old (deprecated) name for `axis`.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If `predictions` shape doesn't match `labels` shape, or
      `axis`, `labels`, `predictions` or `weights` is `None`.

  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
    axis = deprecated_argument_lookup('axis', axis, 'dim', dim)
    if axis is None:
        raise ValueError('You must specify argument `axis`.')
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'cosine_distance_loss', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        radial_diffs = math_ops.multiply(predictions, labels)
        losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(axis,), keepdims=True)
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)