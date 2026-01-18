from tensorflow.python import pywrap_tfe
def start_cancel(self):
    """Cancels blocking operations that have been registered with this object."""
    pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)