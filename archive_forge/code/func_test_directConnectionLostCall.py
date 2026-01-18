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
def test_directConnectionLostCall(self):
    """
        If C{connectionLost} is called directly on a port object, it succeeds
        (and doesn't expect the presence of a C{deferred} attribute).

        C{connectionLost} is called by L{reactor.disconnectAll} at shutdown.
        """
    serverFactory = MyServerFactory()
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    portNumber = port.getHost().port
    port.connectionLost(None)
    client = MyClientFactory()
    serverFactory.protocolConnectionMade = defer.Deferred()
    client.protocolConnectionMade = defer.Deferred()
    reactor.connectTCP('127.0.0.1', portNumber, client)

    def check(ign):
        client.reason.trap(error.ConnectionRefusedError)
    return client.failDeferred.addCallback(check)