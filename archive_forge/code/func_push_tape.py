from tensorflow.python import pywrap_tfe
def push_tape(tape):
    """Pushes an existing tape onto the tape stack."""
    pywrap_tfe.TFE_Py_TapeSetAdd(tape._tape)