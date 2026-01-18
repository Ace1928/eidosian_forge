import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def reportDisconnect(self):
    if self.tls is not None:
        err = NativeOpenSSLError()
    else:
        err = self.disconnectReason
    self.protocol.connectionLost(Failure(err))