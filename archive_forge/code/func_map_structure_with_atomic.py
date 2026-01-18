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
def map_structure_with_atomic(is_atomic_fn, map_fn, nested):
    """Maps the atomic elements of a nested structure.

  Args:
    is_atomic_fn: A function that determines if an element of `nested` is
      atomic.
    map_fn: The function to apply to atomic elements of `nested`.
    nested: A nested structure.

  Returns:
    The nested structure, with atomic elements mapped according to `map_fn`.

  Raises:
    ValueError: If an element that is neither atomic nor a sequence is
      encountered.
  """
    if is_atomic_fn(nested):
        return map_fn(nested)
    if not nest.is_nested(nested):
        raise ValueError('Received non-atomic and non-sequence element: {}'.format(nested))
    if nest.is_mapping(nested):
        values = [nested[k] for k in sorted(nested.keys())]
    elif nest.is_attrs(nested):
        values = _astuple(nested)
    else:
        values = nested
    mapped_values = [map_structure_with_atomic(is_atomic_fn, map_fn, ele) for ele in values]
    return nest._sequence_like(nested, mapped_values)