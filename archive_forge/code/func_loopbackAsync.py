import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
def loopbackAsync(server, client, pumpPolicy=identityPumpPolicy):
    """
    Establish a connection between C{server} and C{client} then transfer data
    between them until the connection is closed. This is often useful for
    testing a protocol.

    @param server: The protocol instance representing the server-side of this
        connection.

    @param client: The protocol instance representing the client-side of this
        connection.

    @param pumpPolicy: When either C{server} or C{client} writes to its
        transport, the string passed in is added to a queue of data for the
        other protocol.  Eventually, C{pumpPolicy} will be called with one such
        queue and the corresponding protocol object.  The pump policy callable
        is responsible for emptying the queue and passing the strings it
        contains to the given protocol's C{dataReceived} method.  The signature
        of C{pumpPolicy} is C{(queue, protocol)}.  C{queue} is an object with a
        C{get} method which will return the next string written to the
        transport, or L{None} if the transport has been disconnected, and which
        evaluates to C{True} if and only if there are more items to be
        retrieved via C{get}.

    @return: A L{Deferred} which fires when the connection has been closed and
        both sides have received notification of this.
    """
    serverToClient = _LoopbackQueue()
    clientToServer = _LoopbackQueue()
    server.makeConnection(_LoopbackTransport(serverToClient))
    client.makeConnection(_LoopbackTransport(clientToServer))
    return _loopbackAsyncBody(server, serverToClient, client, clientToServer, pumpPolicy)