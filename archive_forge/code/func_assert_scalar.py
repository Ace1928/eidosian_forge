import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['debugging.assert_scalar', 'assert_scalar'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_scalar')
def assert_scalar(tensor, name=None, message=None):
    """Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  Args:
    tensor: A `Tensor`.
    name:  A name for this operation. Defaults to "assert_scalar"
    message: A string to prefix to the default message.

  Returns:
    The input tensor (potentially converted to a `Tensor`).

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  """
    with ops.name_scope(name, 'assert_scalar', [tensor]) as name_scope:
        tensor = ops.convert_to_tensor(tensor, name=name_scope)
        shape = tensor.get_shape()
        message = _message_prefix(message)
        if shape.ndims != 0:
            if context.executing_eagerly():
                raise ValueError('%sExpected scalar shape, saw shape: %s.' % (message, shape))
            else:
                raise ValueError('%sExpected scalar shape for %s, saw shape: %s.' % (message, tensor.name, shape))
        return tensor