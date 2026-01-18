from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(array_ops.one_hot)
def ragged_one_hot(indices: ragged_tensor.Ragged, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None):
    """Applies tf.one_hot along the values of a RaggedTensor."""
    if isinstance(axis, int) and axis >= 0:
        if axis <= indices.ragged_rank:
            raise ValueError('axis (%d) must be greater than indices.ragged_rank (%d).' % (axis, indices.ragged_rank))
        axis -= indices.ragged_rank
    with ops.name_scope(name, 'RaggedOneHot', [indices, depth, on_value, off_value, axis]):
        indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(indices, name='indices')
        return indices.with_flat_values(array_ops.one_hot(indices.flat_values, depth, on_value, off_value, axis, dtype, name))