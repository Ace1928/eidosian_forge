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
@tf_export(v1=['metrics.specificity_at_sensitivity'])
def specificity_at_sensitivity(labels, predictions, sensitivity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None):
    """Computes the specificity at a given sensitivity.

  The `specificity_at_sensitivity` function creates four local
  variables, `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` that are used to compute the specificity at the given
  sensitivity value. The threshold for the given sensitivity value is computed
  and used to evaluate the corresponding specificity.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `specificity`. `update_op` increments the `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` counts with the weight of each case
  found in the `predictions` and `labels`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Args:
    labels: The ground truth values, a `Tensor` whose dimensions must match
      `predictions`. Will be cast to `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    sensitivity: A scalar value in range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use for matching the given
      sensitivity.
    metrics_collections: An optional list of collections that `specificity`
      should be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    specificity: A scalar `Tensor` representing the specificity at the given
      `sensitivity` value.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `specificity`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `sensitivity` is not between 0 and 1, or if either `metrics_collections`
      or `updates_collections` are not a list or tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.specificity_at_sensitivity is not supported when eager execution is enabled.')
    if sensitivity < 0 or sensitivity > 1:
        raise ValueError(f'`sensitivity` must be in the range [0, 1]. Currently, `sensitivity` is {sensitivity}.')
    with variable_scope.variable_scope(name, 'specificity_at_sensitivity', (predictions, labels, weights)):
        kepsilon = 1e-07
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]
        values, update_ops = _confusion_matrix_at_thresholds(labels, predictions, thresholds, weights)

        def compute_specificity_at_sensitivity(tp, tn, fp, fn, name):
            """Computes the specificity at the given sensitivity.

      Args:
        tp: True positives.
        tn: True negatives.
        fp: False positives.
        fn: False negatives.
        name: The name of the operation.

      Returns:
        The specificity using the aggregated values.
      """
            sensitivities = math_ops.divide(tp, tp + fn + kepsilon)
            min_val = math_ops.reduce_min(math_ops.abs(sensitivities - sensitivity))
            indices_at_minval = math_ops.equal(math_ops.abs(sensitivities - sensitivity), min_val)
            indices_at_minval = math_ops.cast(indices_at_minval, dtypes.int64)
            indices_at_minval = math_ops.cumsum(indices_at_minval)
            tf_index = math_ops.argmax(indices_at_minval, 0)
            tf_index = math_ops.cast(tf_index, dtypes.int32)
            return math_ops.divide(tn[tf_index], tn[tf_index] + fp[tf_index] + kepsilon, name)

        def specificity_across_replicas(_, values):
            return compute_specificity_at_sensitivity(values['tp'], values['tn'], values['fp'], values['fn'], 'value')
        specificity = _aggregate_across_replicas(metrics_collections, specificity_across_replicas, values)
        update_op = compute_specificity_at_sensitivity(update_ops['tp'], update_ops['tn'], update_ops['fp'], update_ops['fn'], 'update_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        return (specificity, update_op)