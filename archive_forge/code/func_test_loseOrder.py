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
def test_loseOrder(self):
    """
        Check that Protocol.connectionLost is called before factory's
        clientConnectionLost
        """
    server = MyServerFactory()
    server.protocolConnectionMade = defer.Deferred().addCallback(lambda proto: self.addCleanup(proto.transport.loseConnection))
    client = MyClientFactory()
    client.protocolConnectionLost = defer.Deferred()
    client.protocolConnectionMade = defer.Deferred()

    def _cbCM(res):
        """
            protocol.connectionMade callback
            """
        reactor.callLater(0, client.protocol.transport.loseConnection)
    client.protocolConnectionMade.addCallback(_cbCM)
    port = reactor.listenTCP(0, server, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    connector = reactor.connectTCP(port.getHost().host, port.getHost().port, client)
    self.addCleanup(connector.disconnect)

    def _cbCCL(res):
        """
            factory.clientConnectionLost callback
            """
        return 'CCL'

    def _cbCL(res):
        """
            protocol.connectionLost callback
            """
        return 'CL'

    def _cbGather(res):
        self.assertEqual(res, ['CL', 'CCL'])
    d = defer.gatherResults([client.protocolConnectionLost.addCallback(_cbCL), client.deferred.addCallback(_cbCCL)])
    return d.addCallback(_cbGather)