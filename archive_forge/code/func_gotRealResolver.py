from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def gotRealResolver(self, resolver):
    w = self.waiting
    self.__dict__ = resolver.__dict__
    self.__class__ = resolver.__class__
    for d in w:
        d.callback(resolver)