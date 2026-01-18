import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['math.reduce_sum', 'reduce_sum'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'keep_dims is deprecated, use keepdims instead', 'keep_dims')
def reduce_sum_v1(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None):
    """Computes the sum of elements across dimensions of a tensor.

  This is the reduction operation for the elementwise `tf.math.add` op.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  of the entries in `axis`, which must be unique. If `keepdims` is true, the
  reduced dimensions are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
    >>> x.numpy()
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> tf.reduce_sum(x).numpy()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> tf.reduce_sum(x, 0).numpy()
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> tf.reduce_sum(x, 1).numpy()
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> tf.reduce_sum(x, 1, keepdims=True).numpy()
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> tf.reduce_sum(x, [0, 1]).numpy()
    6

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.
    keep_dims: Deprecated alias for `keepdims`.

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  """
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'reduction_indices', reduction_indices)
    keepdims = deprecation.deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    return reduce_sum(input_tensor, axis, keepdims, name)