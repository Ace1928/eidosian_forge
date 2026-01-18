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
def pad_to_bounding_box_internal(image, offset_height, offset_width, target_height, target_width, check_dims):
    """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    check_dims: If True, assert that dimensions are non-negative and in range.
      In multi-GPU distributed settings, assertions can cause program slowdown.
      Setting this parameter to `False` avoids this, resulting in faster speed
      in some situations, with the tradeoff being that some error checking is
      not happening.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative. Not raised if `check_dims` is `False`.
  """
    with ops.name_scope(None, 'pad_to_bounding_box', [image]):
        image = ops.convert_to_tensor(image, name='image')
        is_batch = True
        image_shape = image.get_shape()
        if image_shape.ndims == 3:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = array_ops.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError("'image' (shape %s) must have either 3 or 4 dimensions." % image_shape)
        batch, height, width, depth = _ImageDimensions(image, rank=4)
        after_padding_width = target_width - offset_width - width
        after_padding_height = target_height - offset_height - height
        if check_dims:
            assert_ops = _CheckAtLeast3DImage(image, require_static=False)
            assert_ops += _assert(offset_height >= 0, ValueError, 'offset_height must be >= 0')
            assert_ops += _assert(offset_width >= 0, ValueError, 'offset_width must be >= 0')
            assert_ops += _assert(after_padding_width >= 0, ValueError, 'width must be <= target - offset')
            assert_ops += _assert(after_padding_height >= 0, ValueError, 'height must be <= target - offset')
            image = control_flow_ops.with_dependencies(assert_ops, image)
        paddings = array_ops.reshape(array_ops_stack.stack([0, 0, offset_height, after_padding_height, offset_width, after_padding_width, 0, 0]), [4, 2])
        padded = array_ops.pad(image, paddings)
        padded_shape = [None if _is_tensor(i) else i for i in [batch, target_height, target_width, depth]]
        padded.set_shape(padded_shape)
        if not is_batch:
            padded = array_ops.squeeze(padded, axis=[0])
        return padded