import contextlib
from tensorflow.python import pywrap_tfe
@contextlib.contextmanager
def stop_recording():
    """Stop all gradient recording (backprop and forwardprop)."""
    is_stopped = pywrap_tfe.TFE_Py_TapeSetIsStopped()
    try:
        if not is_stopped:
            pywrap_tfe.TFE_Py_TapeSetStopOnThread()
        yield
    finally:
        if not is_stopped:
            pywrap_tfe.TFE_Py_TapeSetRestartOnThread()