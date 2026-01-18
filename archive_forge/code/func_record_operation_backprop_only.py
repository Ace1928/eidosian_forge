import contextlib
from tensorflow.python import pywrap_tfe
def record_operation_backprop_only(op_type, output_tensors, input_tensors, backward_function):
    """Records the operation on all backward tapes in the stack."""
    pywrap_tfe.TFE_Py_TapeSetRecordOperationBackprop(op_type, output_tensors, input_tensors, backward_function)