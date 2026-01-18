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
@tf_export('sparse.transpose', v1=['sparse.transpose', 'sparse_transpose'])
@deprecation.deprecated_endpoints('sparse_transpose')
def sparse_transpose(sp_input, perm=None, name=None):
    """Transposes a `SparseTensor`.

  Permutes the dimensions according to the value of `perm`.  This is the sparse
  version of `tf.transpose`.

  The returned tensor's dimension `i` will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is the rank
  of the input tensor. Hence, by default, this operation performs a regular
  matrix transpose on 2-D input Tensors.

  For example:

  >>> x = tf.SparseTensor(indices=[[0, 1], [0, 3], [2, 3], [3, 1]],
  ...                     values=[1.1, 2.2, 3.3, 4.4],
  ...                     dense_shape=[4, 5])
  >>> print('x =', tf.sparse.to_dense(x))
  x = tf.Tensor(
  [[0.  1.1 0.  2.2 0. ]
  [0.  0.  0.  0.  0. ]
  [0.  0.  0.  3.3 0. ]
  [0.  4.4 0.  0.  0. ]], shape=(4, 5), dtype=float32)

  >>> x_transpose = tf.sparse.transpose(x)
  >>> print('x_transpose =', tf.sparse.to_dense(x_transpose))
  x_transpose = tf.Tensor(
  [[0.  0.  0.  0. ]
  [1.1 0.  0.  4.4]
  [0.  0.  0.  0. ]
  [2.2 0.  3.3 0. ]
  [0.  0.  0.  0. ]], shape=(5, 4), dtype=float32)

  Equivalently, you could call `tf.sparse.transpose(x, perm=[1, 0])`.  The
  `perm` argument is more useful for n-dimensional tensors where n > 2.

  >>> x = tf.SparseTensor(indices=[[0, 0, 1], [0, 0, 3], [1, 2, 3], [1, 3, 1]],
  ...                     values=[1.1, 2.2, 3.3, 4.4],
  ...                     dense_shape=[2, 4, 5])
  >>> print('x =', tf.sparse.to_dense(x))
  x = tf.Tensor(
  [[[0.  1.1 0.  2.2 0. ]
    [0.  0.  0.  0.  0. ]
    [0.  0.  0.  0.  0. ]
    [0.  0.  0.  0.  0. ]]
  [[0.  0.  0.  0.  0. ]
    [0.  0.  0.  0.  0. ]
    [0.  0.  0.  3.3 0. ]
    [0.  4.4 0.  0.  0. ]]], shape=(2, 4, 5), dtype=float32)

  As above, simply calling `tf.sparse.transpose` will default to `perm=[2,1,0]`.

  To take the transpose of a batch of sparse matrices, where 0 is the batch
  dimension, you would set `perm=[0,2,1]`.

  >>> x_transpose = tf.sparse.transpose(x, perm=[0, 2, 1])
  >>> print('x_transpose =', tf.sparse.to_dense(x_transpose))
  x_transpose = tf.Tensor(
  [[[0.  0.  0.  0. ]
    [1.1 0.  0.  0. ]
    [0.  0.  0.  0. ]
    [2.2 0.  0.  0. ]
    [0.  0.  0.  0. ]]
  [[0.  0.  0.  0. ]
    [0.  0.  0.  4.4]
    [0.  0.  0.  0. ]
    [0.  0.  3.3 0. ]
    [0.  0.  0.  0. ]]], shape=(2, 5, 4), dtype=float32)

  Args:
    sp_input: The input `SparseTensor`.
    perm: A permutation vector of the dimensions of `sp_input`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A transposed `SparseTensor`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
    with ops.name_scope(name, 'SparseTranspose', [sp_input]) as name:
        if perm is None:
            if sp_input.shape.rank is not None:
                rank = sp_input.shape.rank
                perm = rank - 1 - np.arange(0, rank, 1)
            else:
                rank = array_ops.rank(sp_input)
                perm = rank - 1 - math_ops.range(0, rank, 1)
        indices = sp_input.indices
        transposed_indices = array_ops.transpose(array_ops.gather(array_ops.transpose(indices), perm))
        perm_ = tensor_util.constant_value(ops.convert_to_tensor(perm))
        if perm_ is not None and sp_input.get_shape().is_fully_defined():
            old_shape_ = sp_input.get_shape().as_list()
            transposed_dense_shape = list(old_shape_)
            for i, p in enumerate(perm_):
                transposed_dense_shape[i] = old_shape_[p]
        else:
            dense_shape = sp_input.dense_shape
            transposed_dense_shape = array_ops.gather(dense_shape, perm)
        transposed_st = sparse_tensor.SparseTensor(transposed_indices, sp_input.values, transposed_dense_shape)
        transposed_st = sparse_reorder(transposed_st)
        return transposed_st