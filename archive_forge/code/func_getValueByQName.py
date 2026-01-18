from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getValueByQName(self, name):
    for nsname, qname in self._qnames.items():
        if qname == name:
            return self._attrs[nsname]
    raise KeyError(name)