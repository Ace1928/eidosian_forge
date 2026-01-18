from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getQNames(self):
    return list(self._qnames.values())