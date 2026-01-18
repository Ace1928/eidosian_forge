import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated('2018-11-30', 'This op will be removed after the deprecation date. Please switch to tf.sets.difference().')
@tf_export(v1=['setdiff1d'])
@dispatch.add_dispatch_support
def setdiff1d(x, y, index_dtype=dtypes.int32, name=None):
    """Computes the difference between two lists of numbers or strings.

  Given a list x and a list y, this operation returns a list out that
  represents all values that are in x but not in y. The returned list
  out is sorted in the same order that the numbers appear in x
  (duplicates are preserved). This operation also returns a list idx
  that represents the position of each out element in x.

  In other words:

  ```python
  out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]
  ```

  Example usage:

  >>> x = [1, 2, 3, 4, 5, 6]
  >>> y = [1, 3, 5]
  >>> setdiff1d(x,y)
  ListDiff(out=<tf.Tensor: id=2, shape=(3,), dtype=int32,
  numpy=array([2, 4, 6], dtype=int32)>, idx=<tf.Tensor: id=3,
  shape=(3,), dtype=int32, numpy=array([1, 3, 5], dtype=int32)>)

  Args:
    x: A Tensor. 1-D. Values to keep.
    y: A Tensor. Must have the same type as x. 1-D. Values to remove.
    out_idx: An optional tf.DType from: tf.int32, tf.int64. Defaults to
      tf.int32.
    name: A name for the operation (optional).

  Returns:
    A tuple of Tensor objects (out, idx).
    out: A Tensor. Has the same type as x.
    idx: A Tensor of type out_idx.
  """
    return gen_array_ops.list_diff(x, y, index_dtype, name)