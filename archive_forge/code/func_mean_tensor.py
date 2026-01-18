from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['metrics.mean_tensor'])
def mean_tensor(values, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes the element-wise (weighted) mean of the given tensors.

  In contrast to the `mean` function which returns a scalar with the
  mean,  this function returns an average tensor with the same shape as the
  input tensors.

  The `mean_tensor` function creates two local variables,
  `total_tensor` and `count_tensor` that are used to compute the average of
  `values`. This average is ultimately returned as `mean` which is an idempotent
  operation that simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean`.
  `update_op` increments `total` with the reduced sum of the product of `values`
  and `weights`, and it increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    values: A `Tensor` of arbitrary dimensions.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `values`, and must be broadcastable to `values` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `values` dimension).
    metrics_collections: An optional list of collections that `mean`
      should be added to.
    updates_collections: An optional list of collections that `update_op`
      should be added to.
    name: An optional variable_scope name.

  Returns:
    mean: A float `Tensor` representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `mean_value`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match `values`,
      or if either `metrics_collections` or `updates_collections` are not a list
      or tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_tensor is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'mean', (values, weights)):
        values = math_ops.cast(values, dtypes.float32)
        total = metric_variable(values.get_shape(), dtypes.float32, name='total_tensor')
        count = metric_variable(values.get_shape(), dtypes.float32, name='count_tensor')
        num_values = array_ops.ones_like(values)
        if weights is not None:
            values, _, weights = _remove_squeezable_dimensions(predictions=values, labels=None, weights=weights)
            weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtypes.float32), values)
            values = math_ops.multiply(values, weights)
            num_values = math_ops.multiply(num_values, weights)
        update_total_op = state_ops.assign_add(total, values)
        with ops.control_dependencies([values]):
            update_count_op = state_ops.assign_add(count, num_values)
        compute_mean = lambda _, t, c: math_ops.div_no_nan(t, math_ops.maximum(c, 0), name='value')
        mean_t = _aggregate_across_replicas(metrics_collections, compute_mean, total, count)
        update_op = math_ops.div_no_nan(update_total_op, math_ops.maximum(update_count_op, 0), name='update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (mean_t, update_op)