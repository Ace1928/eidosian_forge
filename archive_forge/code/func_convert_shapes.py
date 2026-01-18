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
def convert_shapes(input_shape, to_tuples=True):
    """Converts nested shape representations to desired format.

  Performs:

  TensorShapes -> tuples if `to_tuples=True`.
  tuples of int or None -> TensorShapes if `to_tuples=False`.

  Valid objects to be converted are:
  - TensorShapes
  - tuples with elements of type int or None.
  - ints
  - None

  Args:
    input_shape: A nested structure of objects to be converted to TensorShapes.
    to_tuples: If `True`, converts all TensorShape to tuples. Otherwise converts
      all tuples representing shapes to TensorShapes.

  Returns:
    Nested structure of shapes in desired format.

  Raises:
    ValueError: when the input tensor shape can't be converted to tuples, eg
      unknown tensor shape.
  """

    def _is_shape_component(value):
        return value is None or isinstance(value, (int, tensor_shape.Dimension))

    def _is_atomic_shape(input_shape):
        if _is_shape_component(input_shape):
            return True
        if isinstance(input_shape, tensor_shape.TensorShape):
            return True
        if isinstance(input_shape, (tuple, list)) and all((_is_shape_component(ele) for ele in input_shape)):
            return True
        return False

    def _convert_shape(input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if to_tuples:
            input_shape = tuple(input_shape.as_list())
        return input_shape
    return map_structure_with_atomic(_is_atomic_shape, _convert_shape, input_shape)