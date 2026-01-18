import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def getOutBuffer(self):
    """
        Get the pending writes from this transport, clearing them from the
        pending buffer.

        @return: the bytes written with C{transport.write}
        @rtype: L{bytes}
        """
    S = self.stream
    if S:
        self.stream = []
        return b''.join(S)
    elif self.tls is not None:
        if self.tls.readyToSend:
            self.tls.sent = True
            return self.tls
        else:
            return None
    else:
        return None