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
def test_extraBytesPassedBack(self):
    """
        If extra bytes are received past the end of a response, they are passed
        to the finish callback.
        """
    finished = []
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
    protocol.dataReceived(b'Content-Length: 0\r\n')
    protocol.dataReceived(b'\r\nHere is another thing!')
    self.assertEqual(protocol.state, DONE)
    self.assertEqual(finished, [b'Here is another thing!'])