import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def match_row_splits_dtypes(*tensors, **kwargs):
    """Return a copy of `tensors` with row_splits all having the same dtype.

  Args:
    *tensors: A list of Tensors or RaggedTensors.
    **kwargs: If 'return_dtype=True', then return a tuple (dtype, tensors),
      where `dtype` is the data type used by row-splits, and `tensors` is the
      converted list of `Tensors` and `RaggedTensors`.

  Returns:
    The converted list of `Tensors` and `RaggedTensors`.
  """
    return_dtype = kwargs.pop('return_dtype', False)
    if kwargs:
        raise ValueError(f'Unexpected keyword args {kwargs}.')
    has_int32 = False
    has_int64 = False
    for tensor in tensors:
        if isinstance(tensor, RaggedTensor):
            if tensor.row_splits.dtype == dtypes.int32:
                has_int32 = True
            else:
                has_int64 = True
    if has_int32 and has_int64:
        if not ragged_config.auto_cast_partition_dtype():
            raise ValueError('Input RaggedTensors have mismatched row_splits dtypes; use RaggedTensor.with_row_splits_dtype() to convert them to compatible dtypes.')
        dtype = dtypes.int64
        tensors = tuple((t.with_row_splits_dtype(dtypes.int64) if isinstance(t, RaggedTensor) else t for t in tensors))
    elif has_int32:
        dtype = dtypes.int32
    else:
        dtype = dtypes.int64
    if return_dtype:
        return (dtype, tensors)
    else:
        return tensors