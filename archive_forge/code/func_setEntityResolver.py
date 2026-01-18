from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setEntityResolver(self, resolver):
    """Register an object to resolve external entities."""
    self._ent_handler = resolver