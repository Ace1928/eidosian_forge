import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def failOnce(self, reason=Failure(ConnectionRefusedError())):
    """
        Fail a single TCP connection established on this
        L{ConnectionCompleter}'s L{MemoryReactor}.

        @param reason: the reason to provide that the connection failed.
        @type reason: L{Failure}
        """
    self._reactor.tcpClients.pop(0)[2].clientConnectionFailed(self._reactor.connectors.pop(0), reason)