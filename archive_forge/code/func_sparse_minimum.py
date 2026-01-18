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
@tf_export('sparse.minimum', v1=['sparse.minimum', 'sparse_minimum'])
@deprecation.deprecated_endpoints('sparse_minimum')
def sparse_minimum(sp_a, sp_b, name=None):
    """Returns the element-wise min of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Example:

    >>> sp_zero = tf.sparse.SparseTensor([[0]], [0], [7])
    >>> sp_one = tf.sparse.SparseTensor([[1]], [1], [7])
    >>> res = tf.sparse.minimum(sp_zero, sp_one)
    >>> res.indices
    <tf.Tensor: shape=(2, 1), dtype=int64, numpy=
    array([[0],
           [1]])>
    >>> res.values
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0, 0], dtype=int32)>
    >>> res.dense_shape
    <tf.Tensor: shape=(1,), dtype=int64, numpy=array([7])>

  Args:
    sp_a: a `SparseTensor` operand whose dtype is real, and indices
      lexicographically ordered.
    sp_b: the other `SparseTensor` operand with the same requirements (and the
      same shape).
    name: optional name of the operation.
  Returns:
    output: the output SparseTensor.
  """
    with ops.name_scope(name, 'SparseSparseMinimum', [sp_a.indices, sp_a.values, sp_b.indices, sp_b.values]) as name:
        out_indices, out_values = gen_sparse_ops.sparse_sparse_minimum(sp_a.indices, sp_a.values, sp_a.dense_shape, sp_b.indices, sp_b.values, sp_b.dense_shape, name=name)
    return sparse_tensor.SparseTensor(out_indices, out_values, sp_a.dense_shape)