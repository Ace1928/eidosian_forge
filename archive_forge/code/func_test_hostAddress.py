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
def test_hostAddress(self):
    """
        L{IListeningPort.getHost} returns the same address as a client
        connection's L{ITCPTransport.getPeer}.
        """
    serverFactory = MyServerFactory()
    serverFactory.protocolConnectionLost = defer.Deferred()
    serverConnectionLost = serverFactory.protocolConnectionLost
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    n = port.getHost().port
    clientFactory = MyClientFactory()
    onConnection = clientFactory.protocolConnectionMade = defer.Deferred()
    connector = reactor.connectTCP('127.0.0.1', n, clientFactory)

    def check(ignored):
        self.assertEqual([port.getHost()], clientFactory.peerAddresses)
        self.assertEqual(port.getHost(), clientFactory.protocol.transport.getPeer())
    onConnection.addCallback(check)

    def cleanup(ignored):
        connector.disconnect()
        return serverConnectionLost
    onConnection.addCallback(cleanup)
    return onConnection