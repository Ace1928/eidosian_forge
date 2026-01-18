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
@tf_export('sparse.map_values', v1=[])
@dispatch.add_dispatch_support
def map_values(op, *args, **kwargs):
    """Applies `op` to the `.values` tensor of one or more `SparseTensor`s.

  Replaces any `SparseTensor` in `args` or `kwargs` with its `values`
  tensor (which contains the non-default values for the SparseTensor),
  and then calls `op`.  Returns a `SparseTensor` that is constructed
  from the input `SparseTensor`s' `indices`, `dense_shape`, and the
  value returned by the `op`.

  If the input arguments contain multiple `SparseTensor`s, then they must have
  equal `indices` and dense shapes.

  Examples:

  >>> s = tf.sparse.from_dense([[1, 2, 0],
  ...                           [0, 4, 0],
  ...                           [1, 0, 0]])
  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.ones_like, s)).numpy()
  array([[1, 1, 0],
         [0, 1, 0],
         [1, 0, 0]], dtype=int32)

  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.multiply, s, s)).numpy()
  array([[ 1,  4,  0],
         [ 0, 16,  0],
         [ 1,  0,  0]], dtype=int32)

  >>> tf.sparse.to_dense(tf.sparse.map_values(tf.add, s, 5)).numpy()
  array([[6, 7, 0],
         [0, 9, 0],
         [6, 0, 0]], dtype=int32)

  Note: even though `tf.add(0, 5) != 0`, implicit zeros
  will remain unchanged. However, if the sparse tensor contains any explicit
  zeros, these will be affected by the mapping!

  Args:
    op: The operation that should be applied to the SparseTensor `values`. `op`
      is typically an element-wise operation (such as math_ops.add), but any
      operation that preserves the shape can be used.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.

  Returns:
    A `SparseTensor` whose `indices` and `dense_shape` matches the `indices`
    and `dense_shape` of all input `SparseTensor`s.
  Raises:
    ValueError: If args contains no `SparseTensor`, or if the `indices`
      or `dense_shape`s of the input `SparseTensor`s are not equal.
  """
    sparse_list = []
    inner_args = _replace_sparse_with_values(args, sparse_list)
    inner_kwargs = _replace_sparse_with_values(kwargs, sparse_list)
    if not sparse_list:
        raise ValueError('No SparseTensor in argument list of map_values')
    with ops.control_dependencies(_assert_sparse_compatible(sparse_list)):
        return sparse_tensor.SparseTensor(sparse_list[0].indices, op(*inner_args, **inner_kwargs), sparse_list[0].dense_shape)