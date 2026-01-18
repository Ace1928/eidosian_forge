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
@dispatch.dispatch_for_api(array_ops.shape)
def ragged_shape(input: ragged_tensor.Ragged, name: Optional[str]=None, out_type=dtypes.int32) -> dynamic_ragged_shape.DynamicRaggedShape:
    """Returns the shape of a RaggedTensor.

  Args:
    input: A `RaggedTensor`
    name: A name for the operation (optional).
    out_type: dtype used to encode the shape.

  Returns:
    A `tf.experimental.DynamicRaggedShape`
  """
    with ops.name_scope(name, 'RaggedShape', [input]):
        return dynamic_ragged_shape.DynamicRaggedShape.from_tensor(input, out_type)