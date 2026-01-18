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
@tf_export(v1=['debugging.assert_non_negative', 'assert_non_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_negative')
@_unary_assert_doc('>= 0', 'non-negative')
def assert_non_negative(x, data=None, summarize=None, message=None, name=None):
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_non_negative', [x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if data is None:
            if context.executing_eagerly():
                name = _shape_and_dtype_str(x)
            else:
                name = x.name
            data = [message, 'Condition x >= 0 did not hold element-wise:', 'x (%s) = ' % name, x]
        zero = ops.convert_to_tensor(0, dtype=x.dtype)
        return assert_less_equal(zero, x, data=data, summarize=summarize)