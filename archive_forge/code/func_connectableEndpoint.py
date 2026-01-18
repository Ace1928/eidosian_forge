import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def connectableEndpoint(debug=False):
    """
    Create an endpoint that can be fired on demand.

    @param debug: A flag; whether to dump output from the established
        connection to stdout.
    @type debug: L{bool}

    @return: A client endpoint, and an object that will cause one of the
        L{Deferred}s returned by that client endpoint.
    @rtype: 2-L{tuple} of (L{IStreamClientEndpoint}, L{ConnectionCompleter})
    """
    reactor = MemoryReactorClock()
    clientEndpoint = TCP4ClientEndpoint(reactor, '0.0.0.0', 4321)
    serverEndpoint = TCP4ServerEndpoint(reactor, 4321)
    serverEndpoint.listen(Factory.forProtocol(Protocol))
    return (clientEndpoint, ConnectionCompleter(reactor))