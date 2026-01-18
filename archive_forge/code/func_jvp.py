import functools
import threading
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def jvp(self, primals, unconnected_gradients=UnconnectedGradients.NONE):
    """Fetches the Jacobian-vector product computed for `primals`.

    Note that this method performs no computation, and simply looks up a JVP
    that was already computed (unlike backprop using a `tf.GradientTape`, where
    the computation happens on the call to `tape.gradient`).

    Args:
      primals: A watched Tensor or structure of Tensors to fetch the JVPs for.
      unconnected_gradients: A value which can either hold 'none' or 'zero' and
        alters the value which will be returned if no JVP was computed for
        `primals`. The possible values and effects are detailed in
        'tf.UnconnectedGradients' and it defaults to 'none'.

    Returns:
      Tensors with the same shapes and dtypes as `primals`, or None if no JVP
      is available.
    """
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    if self._accumulator is None:
        raise ValueError('Called jvp() without first tracing anything.')

    def _fetch_jvp(tensor):
        if hasattr(tensor, 'handle'):
            unwrapped_tensor = ops.convert_to_tensor(tensor.handle)
        else:
            unwrapped_tensor = tensor
        result = pywrap_tfe.TFE_Py_ForwardAccumulatorJVP(self._accumulator, unwrapped_tensor)
        if result is None and unconnected_gradients == UnconnectedGradients.ZERO:
            result = array_ops.zeros_like(tensor)
        return result
    return nest.map_structure(_fetch_jvp, primals)