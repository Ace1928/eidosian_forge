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
@tf_export(v1=['image.resize_image_with_pad'])
@dispatch.add_dispatch_support
def resize_image_with_pad_v1(image, target_height, target_width, method=ResizeMethodV1.BILINEAR, align_corners=False):
    """Resizes and pads an image to a target width and height.

  Resizes an image to a target width and height by keeping
  the aspect ratio the same without distortion. If the target
  dimensions don't match the image dimensions, the image
  is resized and then padded with zeroes to match requested
  dimensions.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `resize_images()`
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

    def _resize_fn(im, new_size):
        return resize_images(im, new_size, method, align_corners=align_corners)
    return _resize_image_with_pad_common(image, target_height, target_width, _resize_fn)