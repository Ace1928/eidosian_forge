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
@tf_export('image.crop_to_bounding_box')
@dispatch.add_dispatch_support
def crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width):
    """Crops an `image` to a specified bounding box.

  This op cuts a rectangular bounding box out of `image`. The top-left corner
  of the bounding box is at `offset_height, offset_width` in `image`, and the
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Example Usage:

  >>> image = tf.constant(np.arange(1, 28, dtype=np.float32), shape=[3, 3, 3])
  >>> image[:,:,0] # print the first channel of the 3-D tensor
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
  array([[ 1.,  4.,  7.],
         [10., 13., 16.],
         [19., 22., 25.]], dtype=float32)>
  >>> cropped_image = tf.image.crop_to_bounding_box(image, 0, 0, 2, 2)
  >>> cropped_image[:,:,0] # print the first channel of the cropped 3-D tensor
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[ 1.,  4.],
         [10., 13.]], dtype=float32)>

  Args:
    image: 4-D `Tensor` of shape `[batch, height, width, channels]` or 3-D
      `Tensor` of shape `[height, width, channels]`.
    offset_height: Vertical coordinate of the top-left corner of the bounding
      box in `image`.
    offset_width: Horizontal coordinate of the top-left corner of the bounding
      box in `image`.
    target_height: Height of the bounding box.
    target_width: Width of the bounding box.

  Returns:
    If `image` was 4-D, a 4-D `Tensor` of shape
    `[batch, target_height, target_width, channels]`.
    If `image` was 3-D, a 3-D `Tensor` of shape
    `[target_height, target_width, channels]`.
    It has the same dtype with `image`.

  Raises:
    ValueError: `image` is not a 3-D or 4-D `Tensor`.
    ValueError: `offset_width < 0` or `offset_height < 0`.
    ValueError: `target_width <= 0` or `target_height <= 0`.
    ValueError: `width < offset_width + target_width` or
      `height < offset_height + target_height`.
  """
    with ops.name_scope(None, 'crop_to_bounding_box', [image]):
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
        assert_ops = _CheckAtLeast3DImage(image, require_static=False)
        batch, height, width, depth = _ImageDimensions(image, rank=4)
        assert_ops += _assert(offset_width >= 0, ValueError, 'offset_width must be >= 0.')
        assert_ops += _assert(offset_height >= 0, ValueError, 'offset_height must be >= 0.')
        assert_ops += _assert(target_width > 0, ValueError, 'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError, 'target_height must be > 0.')
        assert_ops += _assert(width >= target_width + offset_width, ValueError, 'width must be >= target + offset.')
        assert_ops += _assert(height >= target_height + offset_height, ValueError, 'height must be >= target + offset.')
        image = control_flow_ops.with_dependencies(assert_ops, image)
        cropped = array_ops.slice(image, array_ops_stack.stack([0, offset_height, offset_width, 0]), array_ops_stack.stack([array_ops.shape(image)[0], target_height, target_width, array_ops.shape(image)[3]]))
        cropped_shape = [None if _is_tensor(i) else i for i in [batch, target_height, target_width, depth]]
        cropped.set_shape(cropped_shape)
        if not is_batch:
            cropped = array_ops.squeeze(cropped, axis=[0])
        return cropped