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
@tf_export('image.pad_to_bounding_box')
@dispatch.add_dispatch_support
def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Usage Example:

  >>> x = [[[1., 2., 3.],
  ...       [4., 5., 6.]],
  ...       [[7., 8., 9.],
  ...       [10., 11., 12.]]]
  >>> padded_image = tf.image.pad_to_bounding_box(x, 1, 1, 4, 4)
  >>> padded_image
  <tf.Tensor: shape=(4, 4, 3), dtype=float32, numpy=
  array([[[ 0.,  0.,  0.],
  [ 0.,  0.,  0.],
  [ 0.,  0.,  0.],
  [ 0.,  0.,  0.]],
  [[ 0.,  0.,  0.],
  [ 1.,  2.,  3.],
  [ 4.,  5.,  6.],
  [ 0.,  0.,  0.]],
  [[ 0.,  0.,  0.],
  [ 7.,  8.,  9.],
  [10., 11., 12.],
  [ 0.,  0.,  0.]],
  [[ 0.,  0.,  0.],
  [ 0.,  0.,  0.],
  [ 0.,  0.,  0.],
  [ 0.,  0.,  0.]]], dtype=float32)>

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  """
    return pad_to_bounding_box_internal(image, offset_height, offset_width, target_height, target_width, check_dims=True)