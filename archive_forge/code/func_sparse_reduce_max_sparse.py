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
@tf_export(v1=['sparse.reduce_max_sparse', 'sparse_reduce_max_sparse'])
@deprecation.deprecated_endpoints('sparse_reduce_max_sparse')
@deprecation.deprecated_args(None, 'keep_dims is deprecated, use keepdims instead', 'keep_dims')
def sparse_reduce_max_sparse(sp_input, axis=None, keepdims=None, reduction_axes=None, keep_dims=None):
    """Computes the max of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_max()`.  In contrast to SparseReduceSum, this Op returns a
  SparseTensor.

  Note: A gradient is not defined for this function, so it can't be used
  in training models that need gradient descent.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keepdims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    sp_input: The SparseTensor to reduce. Should have numeric type.
    axis: The dimensions to reduce; list or scalar. If `None` (the
      default), reduces all dimensions.
    keepdims: If true, retain reduced dimensions with length 1.
    reduction_axes: Deprecated name of axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced SparseTensor.
  """
    keepdims = deprecation.deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'reduction_axes', reduction_axes)
    if keepdims is None:
        keepdims = False
    output_ind, output_val, output_shape = gen_sparse_ops.sparse_reduce_max_sparse(sp_input.indices, sp_input.values, sp_input.dense_shape, math_ops._ReductionDims(sp_input, axis), keepdims)
    return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)