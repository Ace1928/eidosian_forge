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
def convert_inner_node_data(nested, wrap=False):
    """Either wraps or unwraps innermost node data lists in `ListWrapper` objects.

  Args:
    nested: A nested data structure.
    wrap: If `True`, wrap innermost lists in `ListWrapper` objects. If `False`,
      unwraps `ListWrapper` objects into lists.

  Returns:
    Structure of same type as nested, with lists wrapped/unwrapped.
  """

    def _is_serialized_node_data(nested):
        if isinstance(nested, list) and len(nested) in [3, 4] and isinstance(nested[0], str):
            return True
        return False

    def _is_atomic_nested(nested):
        """Returns `True` if `nested` is a list representing node data."""
        if isinstance(nested, ListWrapper):
            return True
        if _is_serialized_node_data(nested):
            return True
        return not nest.is_nested(nested)

    def _convert_object_or_list(nested):
        """Convert b/t `ListWrapper` object and list representations."""
        if wrap:
            if isinstance(nested, ListWrapper):
                return nested
            if _is_serialized_node_data(nested):
                return ListWrapper(nested)
            return nested
        else:
            if isinstance(nested, ListWrapper):
                return nested.as_list()
            return nested
    return map_structure_with_atomic(_is_atomic_nested, _convert_object_or_list, nested)