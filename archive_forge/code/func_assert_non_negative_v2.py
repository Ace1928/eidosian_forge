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
@tf_export('debugging.assert_non_negative', v1=[])
@dispatch.add_dispatch_support
def assert_non_negative_v2(x, message=None, summarize=None, name=None):
    """Assert the condition `x >= 0` holds element-wise.

  This Op checks that `x[i] >= 0` holds for every element of `x`. If `x` is
  empty, this is trivially satisfied.

  If `x` is not >= 0 everywhere, `message`, as well as the first `summarize`
  entries of `x` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to
      "assert_non_negative".

  Returns:
    Op raising `InvalidArgumentError` unless `x` is all non-negative. This can
      be used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x[i] >= 0` is False. The check can be performed immediately during eager
      execution or if `x` is statically known.
  """
    return assert_non_negative(x=x, summarize=summarize, message=message, name=name)