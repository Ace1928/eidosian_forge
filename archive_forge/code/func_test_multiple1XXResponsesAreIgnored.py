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
def test_multiple1XXResponsesAreIgnored(self):
    """
        It is acceptable for multiple 1XX responses to come through, all of
        which get ignored.
        """
    sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
    following200Response = b'HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n'
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(sample103Response + sample103Response + sample103Response + following200Response)
    self.assertEqual(protocol.response.code, 200)
    self.assertEqual(protocol.response.headers, Headers({}))
    self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123']}))
    self.assertEqual(protocol.response.length, 123)