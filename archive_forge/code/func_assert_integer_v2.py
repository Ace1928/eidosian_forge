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
@tf_export('debugging.assert_integer', v1=[])
@dispatch.add_dispatch_support
def assert_integer_v2(x, message=None, name=None):
    """Assert that `x` is of integer dtype.

  If `x` has a non-integer type, `message`, as well as the dtype of `x` are
  printed, and `InvalidArgumentError` is raised.

  This can always be checked statically, so this method returns nothing.

  Args:
    x: A `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to "assert_integer".

  Raises:
    TypeError:  If `x.dtype` is not a non-quantized integer type.
  """
    assert_integer(x=x, message=message, name=name)