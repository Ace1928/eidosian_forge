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
@tf_export('image.crop_and_resize', v1=[])
@dispatch.add_dispatch_support
def crop_and_resize_v2(image, boxes, box_indices, crop_size, method='bilinear', extrapolation_value=0.0, name=None):
    """Extracts crops from the input image tensor and resizes them.

  Extracts crops from the input image tensor and resizes them using bilinear
  sampling or nearest neighbor sampling (possibly with aspect ratio change) to a
  common output size specified by `crop_size`. This is more general than the
  `crop_to_bounding_box` op which extracts a fixed size slice from the input
  image and does not allow resizing or aspect ratio change. The crops occur
  first and then the resize.

  Returns a tensor with `crops` from the input `image` at positions defined at
  the bounding box locations in `boxes`. The cropped boxes are all resized (with
  bilinear or nearest neighbor interpolation) to a fixed
  `size = [crop_height, crop_width]`. The result is a 4-D tensor
  `[num_boxes, crop_height, crop_width, depth]`. The resizing is corner aligned.
  In particular, if `boxes = [[0, 0, 1, 1]]`, the method will give identical
  results to using `tf.compat.v1.image.resize_bilinear()` or
  `tf.compat.v1.image.resize_nearest_neighbor()`(depends on the `method`
  argument) with
  `align_corners=True`.

  Args:
    image: A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is
      specified in normalized coordinates `[y1, x1, y2, x2]`. A normalized
      coordinate value of `y` is mapped to the image coordinate at `y *
      (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1]` in image height coordinates.
      We do allow `y1` > `y2`, in which case the sampled crop is an up-down
      flipped version of the original image. The width dimension is treated
      similarly. Normalized coordinates outside the `[0, 1]` range are allowed,
      in which case we use `extrapolation_value` to extrapolate the input image
      values.
    box_indices: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0,
      batch)`. The value of `box_ind[i]` specifies the image that the `i`-th box
      refers to.
    crop_size: A 1-D tensor of 2 elements, `size = [crop_height, crop_width]`.
      All cropped image patches are resized to this size. The aspect ratio of
      the image content is not preserved. Both `crop_height` and `crop_width`
      need to be positive.
    method: An optional string specifying the sampling method for resizing. It
      can be either `"bilinear"` or `"nearest"` and default to `"bilinear"`.
      Currently two sampling methods are supported: Bilinear and Nearest
        Neighbor.
    extrapolation_value: An optional `float`. Defaults to `0.0`. Value used for
      extrapolation, when applicable.
    name: A name for the operation (optional).

  Returns:
    A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.

  Usage example:

  >>> BATCH_SIZE = 1
  >>> NUM_BOXES = 5
  >>> IMAGE_HEIGHT = 256
  >>> IMAGE_WIDTH = 256
  >>> CHANNELS = 3
  >>> CROP_SIZE = (24, 24)

  >>> image = tf.random.normal(shape=(
  ...   BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS) )
  >>> boxes = tf.random.uniform(shape=(NUM_BOXES, 4))
  >>> box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0,
  ...   maxval=BATCH_SIZE, dtype=tf.int32)
  >>> output = tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
  >>> output.shape
  TensorShape([5, 24, 24, 3])

  Example with linear interpolation:

  >>> image = np.arange(0, 18, 2).astype('float32').reshape(3, 3)
  >>> result = tf.image.crop_and_resize(
  ...   image[None, :, :, None],
  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method='bilinear')
  >>> result[0][:, :, 0]
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[ 8.,  9., 10.],
           [11., 12., 13.],
           [14., 15., 16.]], dtype=float32)>

  Example with nearest interpolation:

  >>> image = np.arange(0, 18, 2).astype('float32').reshape(3, 3)
  >>> result = tf.image.crop_and_resize(
  ...   image[None, :, :, None],
  ...   np.asarray([[0.5,0.5,1,1]]), [0], [3, 3], method='nearest')
  >>> result[0][:, :, 0]
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[ 8., 10., 10.],
           [14., 16., 16.],
           [14., 16., 16.]], dtype=float32)>


  """
    return gen_image_ops.crop_and_resize(image, boxes, box_indices, crop_size, method, extrapolation_value, name)