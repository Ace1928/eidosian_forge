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
@tf_export('math.log_sigmoid', v1=['math.log_sigmoid', 'log_sigmoid'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('log_sigmoid')
def log_sigmoid(x, name=None):
    """Computes log sigmoid of `x` element-wise.

  Specifically, `y = log(1 / (1 + exp(-x)))`.  For numerical stability,
  we use `y = -tf.nn.softplus(-x)`.

  Args:
    x: A Tensor with type `float32` or `float64`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x`.

  Usage Example:

  If a positive number is large, then its log_sigmoid will approach to 0 since
  the formula will be `y = log( <large_num> / (1 + <large_num>) )` which
  approximates to `log (1)` which is 0.

  >>> x = tf.constant([0.0, 1.0, 50.0, 100.0])
  >>> tf.math.log_sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([-6.9314718e-01, -3.1326169e-01, -1.9287499e-22, -0.0000000e+00],
        dtype=float32)>

  If a negative number is large, its log_sigmoid will approach to the number
  itself since the formula will be `y = log( 1 / (1 + <large_num>) )` which is
  `log (1) - log ( (1 + <large_num>) )` which approximates to `- <large_num>`
  that is the number itself.

  >>> x = tf.constant([-100.0, -50.0, -1.0, 0.0])
  >>> tf.math.log_sigmoid(x)
  <tf.Tensor: shape=(4,), dtype=float32, numpy=
  array([-100.       ,  -50.       ,   -1.3132616,   -0.6931472],
        dtype=float32)>
  """
    with ops.name_scope(name, 'LogSigmoid', [x]) as name:
        x = ops.convert_to_tensor(x, name='x')
        return gen_math_ops.neg(gen_nn_ops.softplus(-x), name=name)