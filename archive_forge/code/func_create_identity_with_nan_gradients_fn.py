from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def create_identity_with_nan_gradients_fn(have_nan_gradients):
    """Returns a function that optionally has NaN gradients.

  This serves as a hook to introduce NaN gradients to a model. This returns an
  identity function. The identity's gradient function will check if the boolean
  tensor `have_nan_gradients` is True. If so, the gradient will be NaN.
  Otherwise, the gradient will also be the identity.

  Args:
    have_nan_gradients: A scalar boolean tensor. If True, gradients will be NaN.
      Otherwise, the gradient function is the identity function.

  Returns:
    An identity function whose gradient function will return NaNs, if
    `have_nan_gradients` is True.
  """

    @custom_gradient.custom_gradient
    def _identity_with_nan_gradients(x):
        """Function whose gradient is NaN iff `have_nan_gradients` is True."""
        x = array_ops.identity(x)

        def grad(dx):
            return cond.cond(have_nan_gradients, lambda: dx * float('NaN'), lambda: dx)
        return (x, grad)

    def identity_with_nan_gradients(x):
        return _identity_with_nan_gradients(x)
    return identity_with_nan_gradients