import numpy as np
from tensorboard.plugins.pr_curve import metadata
def streaming_op(name, labels, predictions, num_thresholds=None, weights=None, metrics_collections=None, updates_collections=None, display_name=None, description=None):
    """Computes a precision-recall curve summary across batches of data.

    This function is similar to op() above, but can be used to compute the PR
    curve across multiple batches of labels and predictions, in the same style
    as the metrics found in tf.metrics.

    This function creates multiple local variables for storing true positives,
    true negative, etc. accumulated over each batch of data, and uses these local
    variables for computing the final PR curve summary. These variables can be
    updated with the returned update_op.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      labels: The ground truth values, a `Tensor` whose dimensions must match
        `predictions`. Will be cast to `bool`.
      predictions: A floating point `Tensor` of arbitrary shape and whose values
        are in the range `[0, 1]`.
      num_thresholds: The number of evenly spaced thresholds to generate for
        computing the PR curve. Defaults to 201.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
      metrics_collections: An optional list of collections that `auc` should be
        added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.

    Returns:
      pr_curve: A string `Tensor` containing a single value: the
        serialized PR curve Tensor summary. The summary contains a
        float32 `Tensor` of dimension (6, num_thresholds). The first
        dimension (of length 6) is of the order: true positives, false
        positives, true negatives, false negatives, precision, recall.
      update_op: An operation that updates the summary with the latest data.
    """
    import tensorflow.compat.v1 as tf
    if num_thresholds is None:
        num_thresholds = _DEFAULT_NUM_THRESHOLDS
    thresholds = [i / float(num_thresholds - 1) for i in range(num_thresholds)]
    with tf.name_scope(name, values=[labels, predictions, weights]):
        tp, update_tp = tf.metrics.true_positives_at_thresholds(labels=labels, predictions=predictions, thresholds=thresholds, weights=weights)
        fp, update_fp = tf.metrics.false_positives_at_thresholds(labels=labels, predictions=predictions, thresholds=thresholds, weights=weights)
        tn, update_tn = tf.metrics.true_negatives_at_thresholds(labels=labels, predictions=predictions, thresholds=thresholds, weights=weights)
        fn, update_fn = tf.metrics.false_negatives_at_thresholds(labels=labels, predictions=predictions, thresholds=thresholds, weights=weights)

        def compute_summary(tp, fp, tn, fn, collections):
            precision = tp / tf.maximum(_MINIMUM_COUNT, tp + fp)
            recall = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)
            return _create_tensor_summary(name, tp, fp, tn, fn, precision, recall, num_thresholds, display_name, description, collections)
        pr_curve = compute_summary(tp, fp, tn, fn, metrics_collections)
        update_op = tf.group(update_tp, update_fp, update_tn, update_fn)
        if updates_collections:
            for collection in updates_collections:
                tf.add_to_collection(collection, update_op)
        return (pr_curve, update_op)