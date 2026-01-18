from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
def stillPaging(self):
    """
        (internal) Method called by Broker.
        """
    if not self._stillPaging:
        self.collector.callRemote('endedPaging', pbanswer=False)
        if self.callback is not None:
            self.callback(*self.callbackArgs, **self.callbackKeyword)
    return self._stillPaging