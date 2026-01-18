import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
def register_symbolic_tensor_type(cls):
    """Allows users to specify types regarded as symbolic `Tensor`s.

  Used in conjunction with `tf.register_tensor_conversion_function`, calling
  `tf.keras.__internal__.utils.register_symbolic_tensor_type(cls)`
  allows non-`Tensor` objects to be plumbed through Keras layers.

  Example:

  ```python
  # One-time setup.
  class Foo(object):
    def __init__(self, input_):
      self._input = input_
    def value(self):
      return tf.constant(42.)

  tf.register_tensor_conversion_function(
      Foo, lambda x, *args, **kwargs: x.value())

  tf.keras.__internal__.utils.register_symbolic_tensor_type(Foo)

  # User-land.
  layer = tf.keras.layers.Lambda(lambda input_: Foo(input_))
  ```

  Args:
    cls: A `class` type which shall be regarded as a symbolic `Tensor`.
  """
    global _user_convertible_tensor_types
    if cls not in _user_convertible_tensor_types:
        keras_tensor.register_keras_tensor_specialization(cls, keras_tensor.UserRegisteredTypeKerasTensor)
    _user_convertible_tensor_types.add(cls)