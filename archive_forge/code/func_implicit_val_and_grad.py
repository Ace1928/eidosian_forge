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
def implicit_val_and_grad(f):
    """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the value and the gradient of f when called with
  the same arguments. The gradient is with respect to all trainable TFE
  variables accessed by `f`.

  This function is useful when the exact set of variables to differentiate with
  is not known ahead of time.

  Example:

  ```python
  dense_layer = tf.compat.v1.layers.Dense(1)
  def loss(x, y):
    return tf.reduce_sum(tf.square(dense_layer(x) - y))

  # Obtain the gradient function.
  val_grad_fn = tfe.implicit_value_and_gradients(loss)

  # Invoke the gradient function with concrete values of x and y.
  x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  y = tf.constant([[10.0], [20.0]])
  value, grads_and_vars = val_grad_fn(x, y)
  print('Value of loss: %s' % value)

  # Apply the gradients to Variables.
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  optimizer.apply_gradients(grads_and_vars)
  ```

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar.

  Returns:
    A function which, when called, returns a tuple pair.
    Its first element is the value to which the function evaluates.
    Its second element is list of (gradient, variable) pairs.

  Raises:
    ValueError: if `f` returns None.
  """

    def grad_fn(*args, **kwds):
        """Computes the gradient of the wrapped function."""
        this_tape = tape.push_new_tape()
        try:
            end_node = f(*args, **kwds)
            if end_node is None:
                raise ValueError('Cannot differentiate a function that returns None; did you forget to return a value from {}?'.format(f.__name__))
        finally:
            tape.pop_tape(this_tape)
        variables = this_tape.watched_variables()
        if not variables:
            raise ValueError('No trainable variables were accessed while the function was being computed.')
        sources = [v.handle for v in variables]
        for s in sources:
            if getattr(s, 'is_packed', False):
                raise ValueError('GradientTape.gradient is not supported on packed EagerTensors yet.')
        grad = imperative_grad.imperative_grad(this_tape, nest.flatten(end_node), sources)
        return (end_node, list(zip(grad, variables)))
    return grad_fn