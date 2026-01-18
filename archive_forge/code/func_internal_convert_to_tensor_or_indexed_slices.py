import collections
import warnings
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def internal_convert_to_tensor_or_indexed_slices(value, dtype=None, name=None, as_ref=False):
    """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
    if isinstance(value, ops.EagerTensor) and (not context.executing_eagerly()):
        return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)
    elif isinstance(value, internal.NativeObject):
        if dtype and (not dtypes.as_dtype(dtype).is_compatible_with(value.dtype)):
            raise ValueError(f'Incompatible tensor conversion requested to `dtype` {dtypes.as_dtype(dtype).name} for `value` ({value}) with dtype {value.dtype.name}.')
        return value
    else:
        return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)