import collections
from tensorflow.python import pywrap_tfe
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util import compat
def imperative_grad(tape, target, sources, output_gradients=None, sources_raw=None, unconnected_gradients=UnconnectedGradients.NONE):
    """Computes gradients from the imperatively defined tape on top of the stack.

  Works by filtering the tape, computing how many downstream usages are of each
  tensor and entry, and repeatedly applying backward functions until we have
  gradients for all sources.

  Args:
   tape: the gradient tape which stores the trace.
   target: either a Tensor or list of Tensors to be differentiated.
   sources: list of Tensors for which we want gradients
   output_gradients: if not None, a list of gradient provided for each Target,
    or None if we are to use the target's computed downstream gradient.
   sources_raw: if not None, a list of the source python objects from which the
    sources were generated. Should have the same length as sources. Only needs
    to be populated if unconnected_gradients is 'zero'.
   unconnected_gradients: determines the value returned if the target and
    sources are unconnected. When 'none' the value returned is None wheras when
    'zero' a zero tensor in the same shape as the sources is returned.

  Returns:
   the gradient wrt each of the sources.

  Raises:
    ValueError: if the arguments are invalid.
    RuntimeError: if something goes wrong.
  """
    try:
        unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    except ValueError:
        raise ValueError('Unknown value for unconnected_gradients: %r' % unconnected_gradients)
    return pywrap_tfe.TFE_Py_TapeGradient(tape._tape, target, sources, output_gradients, sources_raw, compat.as_str(unconnected_gradients.value))