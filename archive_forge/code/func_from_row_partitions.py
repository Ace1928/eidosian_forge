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
@classmethod
def from_row_partitions(cls, row_partitions, dtype=None):
    """Create a shape from row_partitions.

    Args:
      row_partitions: a nonempty list of RowPartition objects.
      dtype: the dtype to use, or None to use the row_partitions dtype.

    Returns:
      a DynamicRaggedShape with inner_rank==1.
    """
    if not row_partitions:
        raise ValueError('row_partitions cannot be empty')
    inner_shape = [row_partitions[-1].nvals()]
    return DynamicRaggedShape(row_partitions, inner_shape, dtype=dtype)