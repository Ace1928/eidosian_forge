from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_loseConnectionWhileProxying(self):
    """
        L{TransportProxyProducer.loseConnection} calls the wrapped transport's
        C{loseConnection}.
        """
    transport = StringTransportWithDisconnection()
    protocol = AccumulatingProtocol()
    protocol.makeConnection(transport)
    transport.protocol = protocol
    proxy = TransportProxyProducer(transport)
    self.assertTrue(transport.connected)
    self.assertEqual(transport.producerState, 'producing')
    proxy.loseConnection()
    self.assertEqual(transport.producerState, 'producing')
    self.assertFalse(transport.connected)