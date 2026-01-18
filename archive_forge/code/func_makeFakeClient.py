import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def makeFakeClient(clientProtocol):
    """
    Create and return a new in-memory transport hooked up to the given protocol.

    @param clientProtocol: The client protocol to use.
    @type clientProtocol: L{IProtocol} provider

    @return: The transport.
    @rtype: L{FakeTransport}
    """
    return FakeTransport(clientProtocol, isServer=False)