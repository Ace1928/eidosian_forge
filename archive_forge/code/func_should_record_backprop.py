import contextlib
from tensorflow.python import pywrap_tfe
def should_record_backprop(tensors):
    """Returns true if any tape in the stack watches any of these tensors.

  Only takes GradientTapes into account, not forward accumulators.

  Args:
    tensors: Tensors to check, typically inputs to an operation.

  Returns:
    Boolean, whether any tape watches any of `tensors`.
  """
    return pywrap_tfe.TFE_Py_TapeSetShouldRecordBackprop(tensors)