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
def apply_indexed_slices_grad(self, grad, local_step=0, name=None):
    """Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    Args:
      grad: The gradient `IndexedSlices` to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    return self.apply_grad(grad_indices=grad.indices, grad_values=grad.values, grad_shape=grad.dense_shape, local_step=local_step, name=name)