import contextlib
from tensorflow.python import pywrap_tfe
def could_possibly_record():
    """Returns True if any tape is active."""
    return not pywrap_tfe.TFE_Py_TapeSetIsEmpty()