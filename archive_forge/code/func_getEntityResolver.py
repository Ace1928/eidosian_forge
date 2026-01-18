from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getEntityResolver(self):
    """Returns the current EntityResolver."""
    return self._ent_handler