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
@tf_export(v1=['sparse.reduce_max', 'sparse_reduce_max'])
@deprecation.deprecated_endpoints('sparse_reduce_max')
@deprecation.deprecated_args(None, 'keep_dims is deprecated, use keepdims instead', 'keep_dims')
@deprecation.deprecated_args(None, 'reduction_axes is deprecated, use axis instead', 'reduction_axes')
def sparse_reduce_max(sp_input, axis=None, keepdims=None, reduction_axes=None, keep_dims=None):
    """Computes `tf.sparse.maximum` of elements across dimensions of a SparseTensor.

  This is the reduction operation for the elementwise `tf.sparse.maximum` op.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  similar to the indexing rules in Python.

  The values not defined in `sp_input` don't participate in the reduce max,
  as opposed to be implicitly assumed 0 -- hence it can return negative values
  for sparse `reduction_axes`. But, in case there are no values in
  `reduction_axes`, it will reduce to 0. See second example below.

  For example:

    # 'x' represents [[1, ?, 2]
    #                 [?, 3, ?]]
    # where ? is implicitly-zero.

    >>> x = tf.sparse.SparseTensor([[0, 0], [0, 2], [1, 1]], [1, 2, 3], [2, 3])
    >>> tf.sparse.reduce_max(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=3>
    >>> tf.sparse.reduce_max(x, 0)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 3, 2], dtype=int32)>
    >>> tf.sparse.reduce_max(x, 1)
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>
    >>> tf.sparse.reduce_max(x, 1, keepdims=True)
    <tf.Tensor: shape=(2, 1), dtype=int32, numpy=
    array([[2],
           [3]], dtype=int32)>
    >>> tf.sparse.reduce_max(x, [0, 1])
    <tf.Tensor: shape=(), dtype=int32, numpy=3>

    # 'y' represents [[-7, ?]
    #                 [ 4, 3]
    #                 [ ?, ?]

    >>> y = tf.sparse.SparseTensor([[0, 0,], [1, 0], [1, 1]], [-7, 4, 3],
    ... [3, 2])
    >>> tf.sparse.reduce_max(y, 1)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([-7,  4,  0], dtype=int32)>

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of `axis`.
    keep_dims:  Deprecated alias for `keepdims`.

  Returns:
    The reduced Tensor.
  """
    keepdims = deprecation.deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'reduction_axes', reduction_axes)
    if keepdims is None:
        keepdims = False
    return gen_sparse_ops.sparse_reduce_max(sp_input.indices, sp_input.values, sp_input.dense_shape, math_ops._ReductionDims(sp_input, axis), keepdims)