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
def broadcast_dynamic_shape_extended(a: DynamicRaggedShape, b: DynamicRaggedShape):
    """Gets the smallest shape to which a and b can broadcast.

  In order to create the smallest shape, one must also do most of the
  work to figure out how to transform from the shapes given. Thus, in addition
  to returning the shape, it also creates transformations from the
  original shapes to the result.

  This is the equivalent of:

  c = broadcast_dynamic_shape(a, b)
  ac = get_broadcaster(a, c)
  bc = get_broadcaster(b, c)
  return (c, ac, bc)

  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
    if a.row_partitions and b.row_partitions:
        if a.dtype != b.dtype:
            raise ValueError("Dtypes don't match")
    elif a.dtype != b.dtype:
        if a.row_partitions:
            b = b.with_dtype(a.dtype)
        elif b.row_partitions:
            a = a.with_dtype(b.dtype)
        else:
            a = a.with_dtype(dtypes.int64)
            b = b.with_dtype(dtypes.int64)
    if a.rank is None or b.rank is None:
        raise ValueError('Unable to broadcast: unknown rank')
    elif a.rank == 0:
        return (b, _Broadcaster(a, b, []), _get_identity_broadcaster(b))
    elif b.rank == 0:
        return (a, _get_identity_broadcaster(a), _Broadcaster(b, a, []))
    elif a.rank == 1 and b.rank == 1:
        [a_layer, b_layer, target] = _broadcast_dynamic_shape_one_layer(a.inner_shape, b.inner_shape)
        target_shape = DynamicRaggedShape._from_inner_shape(target)
        return (target_shape, _Broadcaster(a, target_shape, [a_layer]), _Broadcaster(b, target_shape, [b_layer]))
    if a.rank > b.rank:
        c, bc, ac = _broadcast_dynamic_shape_extended_helper(b, a)
        return (c, ac, bc)
    return _broadcast_dynamic_shape_extended_helper(a, b)