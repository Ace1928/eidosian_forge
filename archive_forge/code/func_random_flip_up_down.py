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
@tf_export('image.random_flip_up_down')
@dispatch.add_dispatch_support
def random_flip_up_down(image, seed=None):
    """Randomly flips an image vertically (upside down).

  With a 1 in 2 chance, outputs the contents of `image` flipped along the first
  dimension, which is `height`.  Otherwise, output the image as-is.
  When passing a batch of images, each image will be randomly flipped
  independent of other images.

  Example usage:

  >>> image = np.array([[[1], [2]], [[3], [4]]])
  >>> tf.image.random_flip_up_down(image, 3).numpy().tolist()
  [[[3], [4]], [[1], [2]]]

  Randomly flip multiple images.

  >>> images = np.array(
  ... [
  ...     [[[1], [2]], [[3], [4]]],
  ...     [[[5], [6]], [[7], [8]]]
  ... ])
  >>> tf.image.random_flip_up_down(images, 4).numpy().tolist()
  [[[[3], [4]], [[1], [2]]], [[[5], [6]], [[7], [8]]]]

  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_flip_up_down`. Unlike using the `seed` param
  with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
  same results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    seed: A Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed` for behavior.

  Returns:
    A tensor of the same type and shape as `image`.
  Raises:
    ValueError: if the shape of `image` not supported.
  """
    random_func = functools.partial(random_ops.random_uniform, seed=seed)
    return _random_flip(image, 0, random_func, 'random_flip_up_down')