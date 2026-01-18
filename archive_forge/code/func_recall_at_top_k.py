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
@tf_export(v1=['metrics.recall_at_top_k'])
def recall_at_top_k(labels, predictions_idx, k=None, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """Computes recall@k of top-k predictions with respect to sparse labels.

  Differs from `recall_at_k` in that predictions must be in the form of top `k`
  class indices, whereas `recall_at_k` expects logits. Refer to `recall_at_k`
  for more details.

  Args:
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
      num_labels=1. N >= 1 and num_labels is the number of target classes for
      the associated prediction. Commonly, N=1 and `labels` has shape
      [batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
      should be in range [0, num_classes), where num_classes is the last
      dimension of `predictions`. Values outside this range always count
      towards `false_negative_at_<k>`.
    predictions_idx: Integer `Tensor` with shape [D1, ... DN, k] where N >= 1.
      Commonly, N=1 and predictions has shape [batch size, k]. The final
      dimension contains the top `k` predicted class indices. [D1, ... DN] must
      match `labels`.
    k: Integer, k for @k metric. Only used for the default op name.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes), where num_classes is the last dimension of
      `predictions`. If class_id is outside this range, the method returns NAN.
    weights: `Tensor` whose rank is either 0, or n-1, where n is the rank of
      `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
      dimensions must be either `1`, or the same as the corresponding `labels`
      dimension).
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `weights` is not `None` and its shape doesn't match
    `predictions`, or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.
  """
    with ops.name_scope(name, _at_k_name('recall', k, class_id=class_id), (predictions_idx, labels, weights)) as scope:
        labels = _maybe_expand_labels(labels, predictions_idx)
        top_k_idx = math_ops.cast(predictions_idx, dtypes.int64)
        tp, tp_update = _streaming_sparse_true_positive_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)
        fn, fn_update = _streaming_sparse_false_negative_at_k(predictions_idx=top_k_idx, labels=labels, k=k, class_id=class_id, weights=weights)

        def compute_recall(_, tp, fn):
            return math_ops.divide(tp, math_ops.add(tp, fn), name=scope)
        metric = _aggregate_across_replicas(metrics_collections, compute_recall, tp, fn)
        update = math_ops.divide(tp_update, math_ops.add(tp_update, fn_update), name='update')
        if updates_collections:
            ops.add_to_collections(updates_collections, update)
        return (metric, update)