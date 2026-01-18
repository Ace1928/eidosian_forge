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
def test_logstrClientSetup(self):
    """
        Check that the log customization of the client transport happens
        once the client is connected.
        """
    server = MyServerFactory()
    client = MyClientFactory()
    client.protocolConnectionMade = defer.Deferred()
    port = reactor.listenTCP(0, server, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    connector = reactor.connectTCP(port.getHost().host, port.getHost().port, client)
    self.addCleanup(connector.disconnect)
    self.assertEqual(connector.transport.logstr, 'Uninitialized')

    def cb(ign):
        self.assertEqual(connector.transport.logstr, 'AccumulatingProtocol,client')
    client.protocolConnectionMade.addCallback(cb)
    return client.protocolConnectionMade