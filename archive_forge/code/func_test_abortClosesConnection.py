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
def test_abortClosesConnection(self):
    """
        L{HTTP11ClientProtocol.abort} will tell the transport to close its
        connection when it is invoked, and returns a C{Deferred} that fires
        when the connection is lost.
        """
    transport = StringTransport()
    protocol = HTTP11ClientProtocol()
    protocol.makeConnection(transport)
    r1 = []
    r2 = []
    protocol.abort().addCallback(r1.append)
    protocol.abort().addCallback(r2.append)
    self.assertEqual((r1, r2), ([], []))
    self.assertTrue(transport.disconnecting)
    protocol.connectionLost(Failure(ConnectionDone()))
    self.assertEqual(r1, [None])
    self.assertEqual(r2, [None])