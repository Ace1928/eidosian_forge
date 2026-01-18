import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
@tf_export('sparse.from_dense')
def from_dense(tensor, name=None):
    """Converts a dense tensor into a sparse tensor.

  Only elements not equal to zero will be present in the result. The resulting
  `SparseTensor` has the same dtype and shape as the input.

  >>> sp = tf.sparse.from_dense([0, 0, 3, 0, 1])
  >>> sp.shape.as_list()
  [5]
  >>> sp.values.numpy()
  array([3, 1], dtype=int32)
  >>> sp.indices.numpy()
  array([[2],
         [4]])

  Args:
    tensor: A dense `Tensor` to be converted to a `SparseTensor`.
    name: Optional name for the op.

  Returns:
    The `SparseTensor`.
  """
    with ops.name_scope(name, 'dense_to_sparse'):
        tensor = ops.convert_to_tensor(tensor)
        indices = array_ops.where_v2(math_ops.not_equal(tensor, array_ops.zeros_like(tensor)))
        values = array_ops.gather_nd(tensor, indices)
        shape = array_ops.shape(tensor, out_type=dtypes.int64)
        return sparse_tensor.SparseTensor(indices, values, shape)