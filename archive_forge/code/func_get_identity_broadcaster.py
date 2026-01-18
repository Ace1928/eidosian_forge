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
def get_identity_broadcaster(cls, nvals, dtype=None):
    """Create an identity broadcaster.

    TODO(martinz): an identity broadcaster can be far more efficient than a
    generic broadcaster. Add an optimized implementation.
    Args:
      nvals: the number of values for the broadcaster.
      dtype: the dtype of the broadcaster, or None to use the dtype of nvals.

    Returns:
      an identity broadcaster from [0....nvals-1] to [0...nvals-1]
    """
    return _GatherLayerBroadcaster(math_ops.range(nvals, dtype=dtype))