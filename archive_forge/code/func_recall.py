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
@tf_export(v1=['metrics.recall'])
def recall(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes the recall of the predictions with respect to the labels.

  The `recall` function creates two local variables, `true_positives`
  and `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives` and `false_negatives`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` that updates these variables and returns the `recall`. `update_op`
  weights each prediction by the corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: The predicted values, a `Tensor` of arbitrary dimensions. Will
      be cast to `bool`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `recall` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: Scalar float `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately and whose value matches
      `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.recall is not supported is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'recall', (predictions, labels, weights)):
        predictions, labels, weights = _remove_squeezable_dimensions(predictions=math_ops.cast(predictions, dtype=dtypes.bool), labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)
        true_p, true_positives_update_op = true_positives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)
        false_n, false_negatives_update_op = false_negatives(labels, predictions, weights, metrics_collections=None, updates_collections=None, name=None)

        def compute_recall(true_p, false_n, name):
            return array_ops.where(math_ops.greater(true_p + false_n, 0), math_ops.divide(true_p, true_p + false_n), 0, name)

        def once_across_replicas(_, true_p, false_n):
            return compute_recall(true_p, false_n, 'value')
        rec = _aggregate_across_replicas(metrics_collections, once_across_replicas, true_p, false_n)
        update_op = compute_recall(true_positives_update_op, false_negatives_update_op, 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (rec, update_op)