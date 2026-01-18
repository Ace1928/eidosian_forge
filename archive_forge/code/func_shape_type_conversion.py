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
def shape_type_conversion(fn):
    """Decorator that handles tuple/TensorShape conversion.

  Used in `compute_output_shape` and `build`.

  Args:
    fn: function to wrap.

  Returns:
    Wrapped function.
  """

    def wrapper(instance, input_shape):
        if input_shape is not None:
            input_shape = convert_shapes(input_shape, to_tuples=True)
        output_shape = fn(instance, input_shape)
        if output_shape is not None:
            output_shape = convert_shapes(output_shape, to_tuples=False)
        return output_shape
    return wrapper