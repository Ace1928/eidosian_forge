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
@tf_export('image.adjust_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def adjust_brightness(image, delta):
    """Adjust the brightness of RGB or Grayscale images.

  This is a convenience method that converts RGB images to float
  representation, adjusts their brightness, and then converts them back to the
  original data type. If several adjustments are chained, it is advisable to
  minimize the number of redundant conversions.

  The value `delta` is added to all components of the tensor `image`. `image` is
  converted to `float` and scaled appropriately if it is in fixed-point
  representation, and `delta` is converted to the same data type. For regular
  images, `delta` should be in the range `(-1,1)`, as it is added to the image
  in floating point representation, where pixel values are in the `[0,1)` range.

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.adjust_brightness(x, delta=0.1)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[ 1.1,  2.1,  3.1],
          [ 4.1,  5.1,  6.1]],
         [[ 7.1,  8.1,  9.1],
          [10.1, 11.1, 12.1]]], dtype=float32)>

  Args:
    image: RGB image or images to adjust.
    delta: A scalar. Amount to add to the pixel values.

  Returns:
    A brightness-adjusted tensor of the same shape and type as `image`.
  """
    with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in [dtypes.float16, dtypes.float32]:
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)
        adjusted = math_ops.add(flt_image, math_ops.cast(delta, flt_image.dtype), name=name)
        return convert_image_dtype(adjusted, orig_dtype, saturate=True)