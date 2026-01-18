import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_connectByService(self):
    """
        L{IReactorTCP.connectTCP} accepts the name of a service instead of a
        port number and connects to the port number associated with that
        service, as defined by L{socket.getservbyname}.
        """
    serverFactory = MyServerFactory()
    serverConnMade = defer.Deferred()
    serverFactory.protocolConnectionMade = serverConnMade
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    portNumber = port.getHost().port
    clientFactory = MyClientFactory()
    clientConnMade = defer.Deferred()
    clientFactory.protocolConnectionMade = clientConnMade

    def fakeGetServicePortByName(serviceName, protocolName):
        if serviceName == 'http' and protocolName == 'tcp':
            return portNumber
        return 10
    self.patch(socket, 'getservbyname', fakeGetServicePortByName)
    reactor.connectTCP('127.0.0.1', 'http', clientFactory)
    connMade = defer.gatherResults([serverConnMade, clientConnMade])

    def connected(result):
        serverProtocol, clientProtocol = result
        self.assertTrue(serverFactory.called, 'Server factory was not called upon to build a protocol.')
        serverProtocol.transport.loseConnection()
        clientProtocol.transport.loseConnection()
    connMade.addCallback(connected)
    return connMade