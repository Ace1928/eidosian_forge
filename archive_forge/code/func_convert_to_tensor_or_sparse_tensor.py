import collections
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['convert_to_tensor_or_sparse_tensor'])
def convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None):
    """Converts value to a `SparseTensor` or `Tensor`.

  Args:
    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
      registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `SparseTensor` or `Tensor` based on `value`.

  Raises:
    RuntimeError: If result type is incompatible with `dtype`.
  """
    if dtype is not None:
        dtype = dtypes.as_dtype(dtype)
    if isinstance(value, SparseTensorValue):
        value = SparseTensor.from_value(value)
    if isinstance(value, SparseTensor):
        if dtype and (not dtype.is_compatible_with(value.dtype)):
            raise RuntimeError(f'Sparse dtype mismatch. Requested: {dtype.name},  Actual: {value.dtype.name}')
        return value
    return ops.convert_to_tensor(value, dtype=dtype, name=name)