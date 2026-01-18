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
@tf_export('image.random_brightness')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def random_brightness(image, max_delta, seed=None):
    """Adjust the brightness of images by a random factor.

  Equivalent to `adjust_brightness()` using a `delta` randomly picked in the
  interval `[-max_delta, max_delta)`.

  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_brightness`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).

  Args:
    image: An image or images to adjust.
    max_delta: float, must be non-negative.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...      [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_brightness(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>

  Returns:
    The brightness-adjusted image(s).

  Raises:
    ValueError: if `max_delta` is negative.
  """
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_brightness(image, delta)