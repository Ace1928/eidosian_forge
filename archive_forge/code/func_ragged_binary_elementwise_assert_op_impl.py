import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_binary_elementwise_assert_apis(ragged_tensor.RaggedOrDense, ragged_tensor.RaggedOrDense)
def ragged_binary_elementwise_assert_op_impl(op, x, y):
    """Binary elementwise assert api handler for RaggedTensors.

  This handles binary assert operations for ragged tensors. Compared with
  `ragged_binary_elementwise_op_impl`, this handler does not compute a ragged
  tensor as output. Instead, it applies the assert operation `op` to input
  tensors based on their ragged shapes and flat_values, and returns the result
  of the assertion operation.

  Args:
    op: a binary assert operation on Tensors.
    x: something that can be coerced to a Tensor or RaggedTensor.
    y: something that can be coerced to a Tensor or RaggedTensor.

  Returns:
    the result of the assertion operation.

  """
    x_is_ragged = ragged_tensor.is_ragged(x)
    y_is_ragged = ragged_tensor.is_ragged(y)
    x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, preferred_dtype=y.dtype if y_is_ragged else None)
    y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, preferred_dtype=x.dtype)
    if x_is_ragged and y_is_ragged:
        x, y = ragged_tensor.match_row_splits_dtypes(x, y)
    if x_is_ragged and y_is_ragged or (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims):
        shape_x = DynamicRaggedShape.from_tensor(x)
        shape_y = DynamicRaggedShape.from_tensor(y)
        if shape_x.dtype != shape_y.dtype:
            if not x_is_ragged:
                shape_x = shape_x.with_dtype(shape_y.dtype)
            elif not y_is_ragged:
                shape_y = shape_y.with_dtype(shape_x.dtype)
        if _row_partitions_identical(shape_x, shape_y):
            return op(x.flat_values, y.flat_values)
        _, bcast_xz, bcast_yz = broadcast_dynamic_shape_extended(shape_x, shape_y)
        x_new_flat = bcast_xz.broadcast_flat_values(x, inner_dimensions=False)
        y_new_flat = bcast_yz.broadcast_flat_values(y, inner_dimensions=False)
        return op(x_new_flat, y_new_flat)
    x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
    y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
    return op(x_values, y_values)