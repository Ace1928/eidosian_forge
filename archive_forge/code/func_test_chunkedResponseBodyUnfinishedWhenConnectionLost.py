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
def test_chunkedResponseBodyUnfinishedWhenConnectionLost(self):
    """
        If the final chunk has not been received when the connection is lost
        (for any reason), the protocol passed to C{deliverBody} has its
        C{connectionLost} method called with a L{Failure} wrapping the
        exception for that reason.
        """
    requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
    self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
    result = []
    requestDeferred.addCallback(result.append)
    response = result[0]
    protocol = AccumulatingProtocol()
    response.deliverBody(protocol)
    self.protocol.dataReceived(b'3\r\nfoo\r\n')
    self.protocol.dataReceived(b'3\r\nbar\r\n')
    self.assertEqual(protocol.data, b'foobar')
    self.protocol.connectionLost(Failure(ArbitraryException()))
    return assertResponseFailed(self, fail(protocol.closedReason), [ArbitraryException, _DataLoss])