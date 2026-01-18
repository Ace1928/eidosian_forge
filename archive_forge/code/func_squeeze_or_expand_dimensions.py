from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def squeeze_or_expand_dimensions(y_pred, y_true=None, sample_weight=None):
    """Squeeze or expand last dimension if needed.

  1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
  (using `confusion_matrix.remove_squeezable_dimensions`).
  2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
  from the new rank of `y_pred`.
  If `sample_weight` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
    y_true: Optional label `Tensor` whose dimensions match `y_pred`.
    sample_weight: Optional weight scalar or `Tensor` whose dimensions match
      `y_pred`.

  Returns:
    Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
    the last dimension squeezed,
    `sample_weight` could be extended by one dimension.
    If `sample_weight` is None, (y_pred, y_true) is returned.
  """
    y_pred_shape = y_pred.shape
    y_pred_rank = y_pred_shape.ndims
    if y_true is not None:
        y_true_shape = y_true.shape
        y_true_rank = y_true_shape.ndims
        if y_true_rank is not None and y_pred_rank is not None:
            if y_pred_rank - y_true_rank != 1 or y_pred_shape[-1] == 1:
                y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(y_true, y_pred)
        else:
            rank_diff = array_ops.rank(y_pred) - array_ops.rank(y_true)
            squeeze_dims = lambda: confusion_matrix.remove_squeezable_dimensions(y_true, y_pred)
            is_last_dim_1 = math_ops.equal(1, array_ops.shape(y_pred)[-1])
            maybe_squeeze_dims = lambda: cond.cond(is_last_dim_1, squeeze_dims, lambda: (y_true, y_pred))
            y_true, y_pred = cond.cond(math_ops.equal(1, rank_diff), maybe_squeeze_dims, squeeze_dims)
    if sample_weight is None:
        return (y_pred, y_true)
    weights_shape = sample_weight.shape
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return (y_pred, y_true, sample_weight)
    if y_pred_rank is not None and weights_rank is not None:
        if weights_rank - y_pred_rank == 1:
            sample_weight = array_ops.squeeze(sample_weight, [-1])
        elif y_pred_rank - weights_rank == 1:
            sample_weight = array_ops.expand_dims(sample_weight, [-1])
        return (y_pred, y_true, sample_weight)
    weights_rank_tensor = array_ops.rank(sample_weight)
    rank_diff = weights_rank_tensor - array_ops.rank(y_pred)
    maybe_squeeze_weights = lambda: array_ops.squeeze(sample_weight, [-1])

    def _maybe_expand_weights():
        expand_weights = lambda: array_ops.expand_dims(sample_weight, [-1])
        return cond.cond(math_ops.equal(rank_diff, -1), expand_weights, lambda: sample_weight)

    def _maybe_adjust_weights():
        return cond.cond(math_ops.equal(rank_diff, 1), maybe_squeeze_weights, _maybe_expand_weights)
    sample_weight = cond.cond(math_ops.equal(weights_rank_tensor, 0), lambda: sample_weight, _maybe_adjust_weights)
    return (y_pred, y_true, sample_weight)