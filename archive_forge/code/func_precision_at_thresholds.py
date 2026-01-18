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
@tf_export(v1=['metrics.precision_at_thresholds'])
def precision_at_thresholds(labels, predictions, thresholds, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes precision values for different `thresholds` on `predictions`.

  The `precision_at_thresholds` function creates four local variables,
  `true_positives`, `true_negatives`, `false_positives` and `false_negatives`
  for various values of thresholds. `precision[i]` is defined as the total
  weight of values in `predictions` above `thresholds[i]` whose corresponding
  entry in `labels` is `True`, divided by the total weight of values in
  `predictions` above `thresholds[i]` (`true_positives[i] / (true_positives[i] +
  false_positives[i])`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: A float `Tensor` of shape `[len(thresholds)]`.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables that
      are used in the computation of `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.precision_at_thresholds is not supported when eager execution is enabled.')
    with variable_scope.variable_scope(name, 'precision_at_thresholds', (predictions, labels, weights)):
        values, update_ops = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights, includes=('tp', 'fp'))
        epsilon = 1e-07

        def compute_precision(tp, fp, name):
            return math_ops.divide(tp, epsilon + tp + fp, name='precision_' + name)

        def precision_across_replicas(_, values):
            return compute_precision(values['tp'], values['fp'], 'value')
        prec = _aggregate_across_replicas(metrics_collections, precision_across_replicas, values)
        update_op = compute_precision(update_ops['tp'], update_ops['fp'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (prec, update_op)