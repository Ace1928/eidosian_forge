from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setErrorHandler(self, handler):
    """Register an object to receive error-message events."""
    self._err_handler = handler