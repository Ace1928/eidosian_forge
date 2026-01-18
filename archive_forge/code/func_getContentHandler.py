from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getContentHandler(self):
    """Returns the current ContentHandler."""
    return self._cont_handler