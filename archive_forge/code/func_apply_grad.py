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
def apply_grad(self, grad_indices, grad_values, grad_shape=None, local_step=0, name=None):
    """Attempts to apply a sparse gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    A sparse gradient is represented by its indices, values and possibly empty
    or None shape. Indices must be a vector representing the locations of
    non-zero entries in the tensor. Values are the non-zero slices of the
    gradient, and must have the same first dimension as indices, i.e., the nnz
    represented by indices and values must be consistent. Shape, if not empty or
    None, must be consistent with the accumulator's shape (if also provided).

    Example:
      A tensor [[0, 0], [0, 1], [2, 3]] can be represented
        indices: [1,2]
        values: [[0,1],[2,3]]
        shape: [3, 2]

    Args:
      grad_indices: Indices of the sparse gradient to be applied.
      grad_values: Values of the sparse gradient to be applied.
      grad_shape: Shape of the sparse gradient to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
    return gen_data_flow_ops.sparse_accumulator_apply_gradient(self._accumulator_ref, local_step=local_step, gradient_indices=math_ops.cast(grad_indices, _dtypes.int64), gradient_values=grad_values, gradient_shape=math_ops.cast([] if grad_shape is None else grad_shape, _dtypes.int64), has_known_shape=grad_shape is not None, name=name)