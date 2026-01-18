from tensorflow.python import pywrap_tfe
def push_new_tape(persistent=False, watch_accessed_variables=True):
    """Pushes a new tape onto the tape stack."""
    tape = pywrap_tfe.TFE_Py_TapeSetNew(persistent, watch_accessed_variables)
    return Tape(tape)