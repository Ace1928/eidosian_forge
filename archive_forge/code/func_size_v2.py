from typing import Sequence
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_types(array_ops.size_v2, StructuredTensor)
def size_v2(input, out_type=None, name=None):
    """Returns the size of a tensor."""
    if out_type is None:
        if flags.config().tf_shape_default_int64.value():
            out_type = dtypes.int64
        else:
            out_type = dtypes.int32
    return size(input, name=name, out_type=out_type)