import collections
import contextlib
from tensorflow.python import pywrap_tfe
def pack_tangents(tensors):
    """Packs forward accumulator state into a TangentInfo tuple.

  Args:
    tensors: A flat list of Tensors to pack forward accumulator state for.

  Returns:
    A tuple of (indices, tangents):
      indices: A sequence of sequences of two-element tuples. Each forward
        accumulator is represented as a sequence of tuples with (primal_index,
        jvp_index). Both integers index into the concatenated `tensors + jvps`
        array.
      tangents: A flat list of Tensors. Best interpreted as a sequence to be
        appended to `tensors`.
  """
    return TangentInfo(*pywrap_tfe.TFE_Py_PackJVPs(tensors))