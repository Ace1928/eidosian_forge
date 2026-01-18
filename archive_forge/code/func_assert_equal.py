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
@tf_export(v1=['debugging.assert_equal', 'assert_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('==', '[1, 2]')
def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
    with ops.name_scope(name, 'assert_equal', [x, y, data]):
        if x is y:
            return None if context.executing_eagerly() else control_flow_ops.no_op()
    return _binary_assert('==', 'assert_equal', math_ops.equal, np.equal, x, y, data, summarize, message, name)