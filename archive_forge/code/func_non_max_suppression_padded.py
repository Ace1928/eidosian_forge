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
@tf_export('image.non_max_suppression_padded')
@dispatch.add_dispatch_support
def non_max_suppression_padded(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), pad_to_max_output_size=False, name=None, sorted_input=False, canonicalized_coordinates=False, tile_size=512):
    """Greedily selects a subset of bounding boxes in descending order of score.

  Performs algorithmically equivalent operation to tf.image.non_max_suppression,
  with the addition of an optional parameter which zero-pads the output to
  be of size `max_output_size`.
  The output of this operation is a tuple containing the set of integers
  indexing into the input collection of bounding boxes representing the selected
  boxes and the number of valid indices in the index set.  The bounding box
  coordinates corresponding to the selected indices can then be obtained using
  the `tf.slice` and `tf.gather` operations.  For example:
    ```python
    selected_indices_padded, num_valid = tf.image.non_max_suppression_padded(
        boxes, scores, max_output_size, iou_threshold,
        score_threshold, pad_to_max_output_size=True)
    selected_indices = tf.slice(
        selected_indices_padded, tf.constant([0]), num_valid)
    selected_boxes = tf.gather(boxes, selected_indices)
    ```

  Args:
    boxes: a tensor of rank 2 or higher with a shape of [..., num_boxes, 4].
      Dimensions except the last two are batch dimensions.
    scores: a tensor of rank 1 or higher with a shape of [..., num_boxes].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression. Note that setting this
      value to a large number may result in OOM error depending on the system
      workload.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IoU (intersection over union).
    score_threshold: a float representing the threshold for box scores. Boxes
      with a score that is not larger than this threshold will be suppressed.
    pad_to_max_output_size: whether to pad the output idx to max_output_size.
      Must be set to True when the input is a batch of images.
    name: name of operation.
    sorted_input: a boolean indicating whether the input boxes and scores
      are sorted in descending order by the score.
    canonicalized_coordinates: if box coordinates are given as
    `[y_min, x_min, y_max, x_max]`, setting to True eliminate redundant
     computation to canonicalize box coordinates.
    tile_size: an integer representing the number of boxes in a tile, i.e.,
      the maximum number of boxes per image that can be used to suppress other
      boxes in parallel; larger tile_size means larger parallelism and
      potentially more redundant work.
  Returns:
    idx: a tensor with a shape of [..., num_boxes] representing the
      indices selected by non-max suppression. The leading dimensions
      are the batch dimensions of the input boxes. All numbers are within
      [0, num_boxes). For each image (i.e., idx[i]), only the first num_valid[i]
      indices (i.e., idx[i][:num_valid[i]]) are valid.
    num_valid: a tensor of rank 0 or higher with a shape of [...]
      representing the number of valid indices in idx. Its dimensions are the
      batch dimensions of the input boxes.
   Raises:
    ValueError: When set pad_to_max_output_size to False for batched input.
  """
    with ops.name_scope(name, 'non_max_suppression_padded'):
        if not pad_to_max_output_size:
            if boxes.get_shape().rank is not None and boxes.get_shape().rank > 2:
                raise ValueError("'pad_to_max_output_size' (value {}) must be True for batched input".format(pad_to_max_output_size))
        if name is None:
            name = ''
        idx, num_valid = non_max_suppression_padded_v2(boxes, scores, max_output_size, iou_threshold, score_threshold, sorted_input, canonicalized_coordinates, tile_size)
        if not pad_to_max_output_size:
            idx = idx[0, :num_valid]
        else:
            batch_dims = array_ops.concat([array_ops.shape(boxes)[:-2], array_ops.expand_dims(max_output_size, 0)], 0)
            idx = array_ops.reshape(idx, batch_dims)
        return (idx, num_valid)