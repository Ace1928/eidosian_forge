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
def num_slices_in_dimension(self, axis):
    """Returns the total number of slices across the indicated dimension."""
    if axis < 0:
        return constant_op.constant(1, dtype=self.dim_size_dtype)
    elif self.is_ragged(axis):
        return math_ops.reduce_sum(self._partitioned_dim_sizes[axis])
    else:
        return self.dimension_size(axis) * self.num_slices_in_dimension(axis - 1)