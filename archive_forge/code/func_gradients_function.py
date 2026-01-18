import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def gradients_function(f, params=None):
    """Returns a function which differentiates f with respect to params.

  Example:
  ```python
  # f(x, y) = (x ^ 3) * y - x * (y ^ 2)
  # Therefore, the 1st order derivatives are:
  #   df / dx = 3 * (x ^ 2) * y - y ^ 2
  #   df / dy = x ^ 3 - 2 * x * y
  # The 2nd order derivatives with respect to x is:
  #   d^2 f / (dx)^2 = 6 * x * y
  def f(x, y):
    return x * x * x * y - x * y * y

  # Obtain a function that returns 1st order gradients.
  grad_fn = tfe.gradients_function(f)

  x = 2.0
  y = 3.0

  # Invoke the 1st order gradient function.
  x_grad, y_grad = grad_fn(x, y)
  assert x_grad.numpy() == 3 * (2 ** 2) * 3 - 3 ** 2
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3

  # Obtain a function that returns the 2nd order gradient with respect to x.
  gradgrad_fn = tfe.gradients_function(lambda x, y: grad_fn(x, y)[0])

  # Invoke the 2nd order gradient function.
  x_gradgrad = gradgrad_fn(x, y)[0]
  assert x_gradgrad.numpy() == 6 * 2 * 3

  # To obtain a callable that returns the gradient(s) of `f` with respect to a
  # subset of its inputs, use the `params` keyword argument with
  # `gradients_function()`.
  ygrad_fn = tfe.gradients_function(f, params=[1])

  (y_grad,) = ygrad_fn(x, y)
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3
  ```

  Note that only tensors with real or complex dtypes are differentiable.

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar. If desired, the tensors can be elementwise multiplied by the
      tensors passed as the `dy` keyword argument to the returned gradient
      function.
    params: list of parameter names of f or list of integers indexing the
      parameters with respect to which we'll differentiate. Passing None
      differentiates with respect to all parameters.

  Returns:
    function which, when called, returns the value of f and the gradient
    of `f` with respect to all of `params`. The function takes an extra optional
    keyword argument `dy`. Setting it allows computation of vector jacobian
    products for vectors other than the vector of ones.

  Raises:
    ValueError: if the params are not all strings or all integers.
  """

    def decorated(*args, **kwds):
        """Computes the gradient of the decorated function."""
        _, grad = val_and_grad_function(f, params=params)(*args, **kwds)
        return grad
    return decorated