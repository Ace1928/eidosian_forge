import itertools
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
def succeedOnce(self, debug=False):
    """
        Complete a single TCP connection established on this
        L{ConnectionCompleter}'s L{MemoryReactor}.

        @param debug: A flag; whether to dump output from the established
            connection to stdout.
        @type debug: L{bool}

        @return: a pump for the connection, or L{None} if no connection could
            be established.
        @rtype: L{IOPump} or L{None}
        """
    memoryReactor = self._reactor
    for clientIdx, clientInfo in enumerate(memoryReactor.tcpClients):
        for serverInfo in memoryReactor.tcpServers:
            factories = _factoriesShouldConnect(clientInfo, serverInfo)
            if factories:
                memoryReactor.tcpClients.remove(clientInfo)
                memoryReactor.connectors.pop(clientIdx)
                clientFactory, serverFactory = factories
                clientProtocol = clientFactory.buildProtocol(None)
                serverProtocol = serverFactory.buildProtocol(None)
                serverTransport = makeFakeServer(serverProtocol)
                clientTransport = makeFakeClient(clientProtocol)
                return connect(serverProtocol, serverTransport, clientProtocol, clientTransport, debug)