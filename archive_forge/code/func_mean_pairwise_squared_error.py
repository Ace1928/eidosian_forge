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
@tf_export(v1=['losses.mean_pairwise_squared_error'])
@dispatch.add_dispatch_support
def mean_pairwise_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES):
    """Adds a pairwise-errors-squared loss to the training procedure.

  Unlike `mean_squared_error`, which is a measure of the differences between
  corresponding elements of `predictions` and `labels`,
  `mean_pairwise_squared_error` is a measure of the differences between pairs of
  corresponding elements of `predictions` and `labels`.

  For example, if `labels`=[a, b, c] and `predictions`=[x, y, z], there are
  three pairs of differences are summed to compute the loss:
    loss = [ ((a-b) - (x-y)).^2 + ((a-c) - (x-z)).^2 + ((b-c) - (y-z)).^2 ] / 3

  Note that since the inputs are of shape `[batch_size, d0, ... dN]`, the
  corresponding pairs are computed within each batch sample but not across
  samples within a batch. For example, if `predictions` represents a batch of
  16 grayscale images of dimension [batch_size, 100, 200], then the set of pairs
  is drawn from each image, but not across images.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector.

  Args:
    labels: The ground truth output tensor, whose shape must match the shape of
      `predictions`.
    predictions: The predicted outputs, a tensor of size
      `[batch_size, d0, .. dN]` where N+1 is the total number of dimensions in
      `predictions`.
    weights: Coefficients for the loss a scalar, a tensor of shape
      `[batch_size]` or a tensor whose shape matches `predictions`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.

  Returns:
    A scalar `Tensor` that returns the weighted loss.

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
    with ops.name_scope(scope, 'mean_pairwise_squared_error', (predictions, labels, weights)) as scope:
        weights = math_ops.cast(weights, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)

        def compute_loss(labels, predictions, weights, loss_collection):
            predictions = math_ops.cast(predictions, dtype=dtypes.float32)
            predictions.get_shape().assert_is_compatible_with(labels.get_shape())
            diffs = math_ops.subtract(predictions, labels)
            axis = math_ops.range(1, array_ops.rank(diffs))
            sum_squares_diff_per_batch = math_ops.reduce_sum(math_ops.square(diffs), axis=axis, keepdims=True)
            num_present_per_batch = _num_present(diffs, weights, per_batch=True)
            term1 = 2.0 * math_ops.div_no_nan(sum_squares_diff_per_batch, math_ops.maximum(num_present_per_batch - 1, 0), name='value')
            sum_diff = math_ops.reduce_sum(diffs, axis=axis, keepdims=True)
            term2 = 2.0 * math_ops.div_no_nan(math_ops.square(sum_diff), math_ops.maximum(math_ops.multiply(num_present_per_batch, num_present_per_batch - 1), 0), name='value')
            weighted_losses = math_ops.multiply(term1 - term2, weights)
            loss = math_ops.reduce_sum(weighted_losses)
            mean_loss = array_ops.where(math_ops.reduce_sum(num_present_per_batch) > 0, loss, array_ops.zeros_like(loss), name='value')
            util.add_loss(mean_loss, loss_collection)
            return mean_loss
        if control_flow_ops.get_enclosing_xla_context() is not None:
            return compute_loss(labels, predictions, weights, loss_collection)
        else:
            with ops.control_dependencies((weights_broadcast_ops.assert_broadcastable(weights, labels),)):
                return compute_loss(labels, predictions, weights, loss_collection)