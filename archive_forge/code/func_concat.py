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
@dispatch.dispatch_for_types(array_ops.concat, StructuredTensor)
def concat(values, axis, name: str='concat'):
    """tf.concat for structured tensors.

  Does not support (yet) checks on illegal axis values, et cetera.

  Args:
    values: a sequence of StructuredTensors.
    axis: an axis to concatenate upon.
    name: the name of the op(s).

  Returns:
    the params reorganized according to indices.
  """
    if name is None:
        name = 'concat'
    _assert_concat_compatible_structured_tensors(values)

    def leaf_op(values):
        return array_ops.concat(values, axis)
    axis = array_ops.get_positive_axis(axis, values[0].rank)
    with ops.name_scope(name, 'StructuredConcat', values):
        return _extend_op(values, leaf_op)