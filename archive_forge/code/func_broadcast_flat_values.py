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
def broadcast_flat_values(self, rt, inner_dimensions=True):
    """flat_values of a ragged tensor broadcast to target_shape.

    If inner_dimensions==True, then the result is a dense tensor with shape
    target_shape.inner_shape, the flat values of the broadcasted shape.

    If you add target_shape.row_partitions, you will get the full broadcasted
    shape.

    If inner_dimensions==False, the result is a dense tensor that satsifies
    certain properties:
    1. broadcast_to(result, target_shape.inner_shape) will give the result
       if inner_dimensions==True.
    2. Either (a) (result.rank < target_shape.inner_rank)
       or (b) (result.shape[0] == target_shape.inner_shape[0]).
    3. result.rank = min(target_shape.inner_rank, rt.rank)
    4. For i < target_shape.inner_rank - 1, and i < rt.rank,
       and if rt.shape[-i]!=1, then result.shape[-i]=target_shape[-i].
    Args:
      rt: a ragged or dense tensor.
      inner_dimensions: if true, broadcast the inner dimensions as well.

    Returns:
      a dense tensor
    """
    if ragged_tensor.is_ragged(rt):
        rt = rt.flat_values
    if self.target_shape.rank == 0:
        return rt
    inner_rank = self.target_shape.inner_rank
    if inner_rank > self._source_shape.rank:
        if self.source_shape.num_row_partitions > 0:
            rt = array_ops.reshape(rt, self.source_shape._alt_inner_shape(self.source_shape.rank))
        if inner_dimensions:
            return array_ops.broadcast_to(rt, self._target_inner_shape_int32())
        return rt
    else:
        if self._source_shape.inner_rank != inner_rank:
            rt = array_ops.reshape(rt, self._source_shape._alt_inner_shape(inner_rank))
        flat_broadcaster = self._layer_broadcasters[-inner_rank]
        rt = flat_broadcaster.broadcast_tensor(rt)
        if inner_dimensions:
            rt = array_ops.broadcast_to(rt, self._target_inner_shape_int32())
        return rt