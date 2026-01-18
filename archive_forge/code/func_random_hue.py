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
@tf_export('image.random_hue')
@dispatch.add_dispatch_support
def random_hue(image, max_delta, seed=None):
    """Adjust the hue of RGB images by a random factor.

  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta)`.

  `max_delta` must be in the interval `[0, 0.5]`.

  Usage Example:

  >>> x = [[[1.0, 2.0, 3.0],
  ...       [4.0, 5.0, 6.0]],
  ...     [[7.0, 8.0, 9.0],
  ...       [10.0, 11.0, 12.0]]]
  >>> tf.image.random_hue(x, 0.2)
  <tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=...>

  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_hue`. Unlike using the `seed` param with
  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
  results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).

  Args:
    image: RGB image or images. The size of the last dimension must be 3.
    max_delta: float. The maximum value for the random delta.
    seed: An operation-specific seed. It will be used in conjunction with the
      graph-level seed to determine the real seeds that will be used in this
      operation. Please see the documentation of set_random_seed for its
      interaction with the graph-level random seed.

  Returns:
    Adjusted image(s), same shape and DType as `image`.

  Raises:
    ValueError: if `max_delta` is invalid.
  """
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')
    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')
    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_hue(image, delta)