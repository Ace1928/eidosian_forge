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
@tf_export('debugging.assert_rank_in', v1=[])
@dispatch.add_dispatch_support
def assert_rank_in_v2(x, ranks, message=None, name=None):
    """Assert that `x` has a rank in `ranks`.

  This Op checks that the rank of `x` is in `ranks`.

  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    ranks: `Iterable` of scalar `Tensor` objects.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_rank_in".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: `x` does not have rank in `ranks`, but the rank cannot
      be statically determined.
    ValueError: If static checks determine `x` has mismatched rank.
  """
    return assert_rank_in(x=x, ranks=ranks, message=message, name=name)