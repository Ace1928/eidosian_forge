import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def numpy(self):
    """Returns a numpy `array` with the values for this `RaggedTensor`.

    Requires that this `RaggedTensor` was constructed in eager execution mode.

    Ragged dimensions are encoded using numpy `arrays` with `dtype=object` and
    `rank=1`, where each element is a single row.

    #### Examples

    In the following example, the value returned by `RaggedTensor.numpy()`
    contains three numpy `array` objects: one for each row (with `rank=1` and
    `dtype=int64`), and one to combine them (with `rank=1` and `dtype=object`):

    >>> tf.ragged.constant([[1, 2, 3], [4, 5]], dtype=tf.int64).numpy()
    array([array([1, 2, 3]), array([4, 5])], dtype=object)

    Uniform dimensions are encoded using multidimensional numpy `array`s.  In
    the following example, the value returned by `RaggedTensor.numpy()` contains
    a single numpy `array` object, with `rank=2` and `dtype=int64`:

    >>> tf.ragged.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64).numpy()
    array([[1, 2, 3], [4, 5, 6]])

    Returns:
      A numpy `array`.
    """
    if not self._is_eager():
        raise ValueError('RaggedTensor.numpy() is only supported in eager mode.')
    values = self.values.numpy()
    splits = self.row_splits.numpy()
    rows = [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
    if not rows:
        return np.zeros((0, 0) + values.shape[1:], dtype=values.dtype)
    has_variable_length_rows = any((len(row) != len(rows[0]) for row in rows))
    dtype = np.object_ if has_variable_length_rows else None
    return np.array(rows, dtype=dtype)