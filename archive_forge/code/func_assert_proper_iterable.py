import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.assert_proper_iterable', v1=['debugging.assert_proper_iterable', 'assert_proper_iterable'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_proper_iterable')
def assert_proper_iterable(values):
    """Static assert that values is a "proper" iterable.

  `Ops` that expect iterables of `Tensor` can call this to validate input.
  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

  Args:
    values:  Object to be checked.

  Raises:
    TypeError:  If `values` is not iterable or is one of
      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
  """
    unintentional_iterables = (tensor_lib.Tensor, sparse_tensor.SparseTensor, np.ndarray) + compat.bytes_or_text_types
    if isinstance(values, unintentional_iterables):
        raise TypeError('Expected argument "values" to be a "proper" iterable.  Found: %s' % type(values))
    if not hasattr(values, '__iter__'):
        raise TypeError('Expected argument "values" to be iterable.  Found: %s' % type(values))