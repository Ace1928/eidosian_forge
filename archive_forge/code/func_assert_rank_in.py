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
@tf_export(v1=['debugging.assert_rank_in', 'assert_rank_in'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_in')
def assert_rank_in(x, ranks, data=None, summarize=None, message=None, name=None):
    """Assert `x` has rank in `ranks`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    ranks:  Iterable of scalar `Tensor` objects.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to "assert_rank_in".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has mismatched rank.
  """
    with ops.name_scope(name, 'assert_rank_in', (x,) + tuple(ranks) + tuple(data or [])):
        if not isinstance(x, sparse_tensor.SparseTensor):
            x = ops.convert_to_tensor(x, name='x')
        ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
        message = _message_prefix(message)
        if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
            name = ''
        else:
            name = x.name
        if data is None:
            data = [message, 'Tensor %s must have rank in' % name] + list(ranks) + ['Received shape: ', array_ops.shape(x)]
        try:
            assert_op = _assert_ranks_condition(x, ranks, _static_rank_in, _dynamic_rank_in, data, summarize)
        except ValueError as e:
            if e.args[0] == 'Static rank condition failed':
                raise ValueError('%sTensor %s must have rank in %s.  Received rank %d, shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
            else:
                raise
    return assert_op