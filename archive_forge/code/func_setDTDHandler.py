from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setDTDHandler(self, handler):
    """Register an object to receive basic DTD-related events."""
    self._dtd_handler = handler