import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('image.non_max_suppression_with_scores')
@dispatch.add_dispatch_support
def non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), soft_nms_sigma=0.0, name=None):
    """Greedily selects a subset of bounding boxes in descending order of score.

  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval `[0, 1]`) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system.  Note that this
  algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is a set of integers indexing into the input
  collection of bounding boxes representing the selected boxes.  The bounding
  box coordinates corresponding to the selected indices can then be obtained
  using the `tf.gather` operation.  For example:
    ```python
    selected_indices, selected_scores = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold=1.0, score_threshold=0.1,
        soft_nms_sigma=0.5)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  This function generalizes the `tf.image.non_max_suppression` op by also
  supporting a Soft-NMS (with Gaussian weighting) mode (c.f.
  Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
  of other overlapping boxes instead of directly causing them to be pruned.
  Consequently, in contrast to `tf.image.non_max_suppression`,
  `tf.image.non_max_suppression_with_scores` returns the new scores of each
  input box in the second output, `selected_scores`.

  To enable this Soft-NMS mode, set the `soft_nms_sigma` parameter to be
  larger than 0.  When `soft_nms_sigma` equals 0, the behavior of
  `tf.image.non_max_suppression_with_scores` is identical to that of
  `tf.image.non_max_suppression` (except for the extra output) both in function
  and in running time.

  Note that when `soft_nms_sigma` > 0, Soft-NMS is performed and `iou_threshold`
  is ignored. `iou_threshold` is only used for standard NMS.

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: A scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non-max suppression.
    iou_threshold: A 0-D float tensor representing the threshold for deciding
      whether boxes overlap too much with respect to IOU.
    score_threshold: A 0-D float tensor representing the threshold for deciding
      when to remove boxes based on score.
    soft_nms_sigma: A 0-D float tensor representing the sigma parameter for Soft
      NMS; see Bodla et al (c.f. https://arxiv.org/abs/1704.04503).  When
      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
      NMS.
    name: A name for the operation (optional).

  Returns:
    selected_indices: A 1-D integer `Tensor` of shape `[M]` representing the
      selected indices from the boxes tensor, where `M <= max_output_size`.
    selected_scores: A 1-D float tensor of shape `[M]` representing the
      corresponding scores for each selected box, where `M <= max_output_size`.
      Scores only differ from corresponding input scores when using Soft NMS
      (i.e. when `soft_nms_sigma>0`)
  """
    with ops.name_scope(name, 'non_max_suppression_with_scores'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(score_threshold, name='score_threshold')
        soft_nms_sigma = ops.convert_to_tensor(soft_nms_sigma, name='soft_nms_sigma')
        selected_indices, selected_scores, _ = gen_image_ops.non_max_suppression_v5(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma, pad_to_max_output_size=False)
        return (selected_indices, selected_scores)