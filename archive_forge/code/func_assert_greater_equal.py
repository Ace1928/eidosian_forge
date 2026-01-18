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
@tf_export(v1=['debugging.assert_greater_equal', 'assert_greater_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_greater_equal')
@_binary_assert_doc('>=', '[1, 0]')
def assert_greater_equal(x, y, data=None, summarize=None, message=None, name=None):
    return _binary_assert('>=', 'assert_greater_equal', math_ops.greater_equal, np.greater_equal, x, y, data, summarize, message, name)