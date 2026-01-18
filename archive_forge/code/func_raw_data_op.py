import numpy as np
from tensorboard.plugins.pr_curve import metadata
def raw_data_op(name, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, num_thresholds=None, display_name=None, description=None, collections=None):
    """Create an op that collects data for visualizing PR curves.

    Unlike the op above, this one avoids computing precision, recall, and the
    intermediate counts. Instead, it accepts those tensors as arguments and
    relies on the caller to ensure that the calculations are correct (and the
    counts yield the provided precision and recall values).

    This op is useful when a caller seeks to compute precision and recall
    differently but still use the PR curves plugin.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      true_positive_counts: A rank-1 tensor of true positive counts. Must contain
          `num_thresholds` elements and be castable to float32. Values correspond
          to thresholds that increase from left to right (from 0 to 1).
      false_positive_counts: A rank-1 tensor of false positive counts. Must
          contain `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      true_negative_counts: A rank-1 tensor of true negative counts. Must contain
          `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      false_negative_counts: A rank-1 tensor of false negative counts. Must
          contain `num_thresholds` elements and be castable to float32. Values
          correspond to thresholds that increase from left to right (from 0 to 1).
      precision: A rank-1 tensor of precision values. Must contain
          `num_thresholds` elements and be castable to float32. Values correspond
          to thresholds that increase from left to right (from 0 to 1).
      recall: A rank-1 tensor of recall values. Must contain `num_thresholds`
          elements and be castable to float32. Values correspond to thresholds
          that increase from left to right (from 0 to 1).
      num_thresholds: Number of thresholds, evenly distributed in `[0, 1]`, to
          compute PR metrics for. Should be `>= 2`. This value should be a
          constant integer value, not a Tensor that stores an integer.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.
      collections: Optional list of graph collections keys. The new
          summary op is added to these collections. Defaults to
          `[Graph Keys.SUMMARIES]`.

    Returns:
      A summary operation for use in a TensorFlow graph. See docs for the `op`
      method for details on the float32 tensor produced by this summary.
    """
    import tensorflow.compat.v1 as tf
    with tf.name_scope(name, values=[true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall]):
        return _create_tensor_summary(name, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, num_thresholds, display_name, description, collections)