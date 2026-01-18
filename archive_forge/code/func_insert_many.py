import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def insert_many(self, component_index, keys, values, name=None):
    """For each key, assigns the respective value to the specified component.

    This operation updates each element at component_index.

    Args:
      component_index: The component of the value that is being assigned.
      keys: A vector of keys, with length n.
      values: An any-dimensional tensor of values, which are associated with the
        respective keys. The first dimension must have length n.
      name: Optional name for the op.

    Returns:
      The operation that performs the insertion.
    Raises:
      InvalidArgumentsError: If inserting keys and values without elements.
    """
    if name is None:
        name = '%s_BarrierInsertMany' % self._name
    return gen_data_flow_ops.barrier_insert_many(self._barrier_ref, keys, values, component_index, name=name)