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
@tf_export(v1=['debugging.assert_near', 'assert_near'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_near')
def assert_near(x, y, rtol=None, atol=None, data=None, summarize=None, message=None, name=None):
    """Assert the condition `x` and `y` are close element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_near(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have

  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.

  If both `x` and `y` are empty, this is trivially satisfied.

  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.

  Args:
    x:  Float or complex `Tensor`.
    y:  Float or complex `Tensor`, same `dtype` as, and broadcastable to, `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to "assert_near".

  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.

  @compatibility(numpy)
  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data
  type. This is due to the fact that `TensorFlow` is often used with `32bit`,
  `64bit`, and even `16bit` data.
  @end_compatibility
  """
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_near', [x, y, rtol, atol, data]):
        x = ops.convert_to_tensor(x, name='x')
        y = ops.convert_to_tensor(y, name='y', dtype=x.dtype)
        dtype = x.dtype
        if dtype.is_complex:
            dtype = dtype.real_dtype
        eps = np.finfo(dtype.as_numpy_dtype).eps
        rtol = 10 * eps if rtol is None else rtol
        atol = 10 * eps if atol is None else atol
        rtol = ops.convert_to_tensor(rtol, name='rtol', dtype=dtype)
        atol = ops.convert_to_tensor(atol, name='atol', dtype=dtype)
        if context.executing_eagerly():
            x_name = _shape_and_dtype_str(x)
            y_name = _shape_and_dtype_str(y)
        else:
            x_name = x.name
            y_name = y.name
        if data is None:
            data = [message, 'x and y not equal to tolerance rtol = %s, atol = %s' % (rtol, atol), 'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y]
        tol = atol + rtol * math_ops.abs(y)
        diff = math_ops.abs(x - y)
        condition = math_ops.reduce_all(math_ops.less(diff, tol))
        return control_flow_assert.Assert(condition, data, summarize=summarize)