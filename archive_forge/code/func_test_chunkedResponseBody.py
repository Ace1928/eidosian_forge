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
def test_chunkedResponseBody(self):
    """
        If the response headers indicate the response body is encoded with the
        I{chunked} transfer encoding, the body is decoded according to that
        transfer encoding before being passed to L{Response._bodyDataReceived}.
        """
    finished = []
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
    body = []
    protocol.response._bodyDataReceived = body.append
    protocol.dataReceived(b'Transfer-Encoding: chunked\r\n')
    protocol.dataReceived(b'\r\n')
    self.assertEqual(body, [])
    self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)
    protocol.dataReceived(b'3\r\na')
    self.assertEqual(body, [b'a'])
    protocol.dataReceived(b'bc\r\n')
    self.assertEqual(body, [b'a', b'bc'])
    protocol.dataReceived(b'0\r\n\r\nextra')
    self.assertEqual(finished, [b'extra'])