from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
def is_ragged(self, axis):
    """Returns true if the indicated dimension is ragged."""
    if not isinstance(axis, int):
        raise TypeError('axis must be an integer')
    rank = self.rank
    if axis < 0:
        raise ValueError('Negative axis values are not supported')
    elif rank is not None and axis >= rank:
        raise ValueError('Expected axis=%s < rank=%s' % (axis, rank))
    else:
        return axis > 0 and axis < len(self._partitioned_dim_sizes) and (self._partitioned_dim_sizes[axis].shape.ndims == 1)