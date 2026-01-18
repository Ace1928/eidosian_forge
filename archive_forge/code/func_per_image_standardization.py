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
@tf_export('image.per_image_standardization')
@dispatch.add_dispatch_support
def per_image_standardization(image):
    """Linearly scales each image in `image` to have mean 0 and variance 1.

  For each 3-D image `x` in `image`, computes `(x - mean) / adjusted_stddev`,
  where

  - `mean` is the average of all values in `x`
  - `adjusted_stddev = max(stddev, 1.0/sqrt(N))` is capped away from 0 to
    protect against division by 0 when handling uniform images
    - `N` is the number of elements in `x`
    - `stddev` is the standard deviation of all values in `x`

  Example Usage:

  >>> image = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
  >>> image # 3-D tensor
  <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
  array([[[ 1,  2,  3],
          [ 4,  5,  6]],
         [[ 7,  8,  9],
          [10, 11, 12]]], dtype=int32)>
  >>> new_image = tf.image.per_image_standardization(image)
  >>> new_image # 3-D tensor with mean ~= 0 and variance ~= 1
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
  array([[[-1.593255  , -1.3035723 , -1.0138896 ],
          [-0.7242068 , -0.4345241 , -0.14484136]],
         [[ 0.14484136,  0.4345241 ,  0.7242068 ],
          [ 1.0138896 ,  1.3035723 ,  1.593255  ]]], dtype=float32)>

  Args:
    image: An n-D `Tensor` with at least 3 dimensions, the last 3 of which are
      the dimensions of each image.

  Returns:
    A `Tensor` with the same shape as `image` and its dtype is `float32`.

  Raises:
    ValueError: The shape of `image` has fewer than 3 dimensions.
  """
    with ops.name_scope(None, 'per_image_standardization', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        image = math_ops.cast(image, dtype=dtypes.float32)
        num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
        image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
        stddev = math_ops.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
        min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
        adjusted_stddev = math_ops.maximum(stddev, min_stddev)
        image -= image_mean
        image = math_ops.divide(image, adjusted_stddev, name=scope)
        return image