from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def setPublicId(self, public_id):
    """Sets the public identifier of this InputSource."""
    self.__public_id = public_id