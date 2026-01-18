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
def test_ignored1XXResponseCausesLog(self):
    """
        When a 1XX response is ignored, Twisted emits a log.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(sample103Response)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    self.assertEquals(event['log_format'], 'Ignoring unexpected {code} response')
    self.assertEquals(event['code'], 103)