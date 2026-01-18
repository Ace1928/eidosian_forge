from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getDTDHandler(self):
    """Returns the current DTD handler."""
    return self._dtd_handler