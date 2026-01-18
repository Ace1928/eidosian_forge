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
def test_connectionLostWithError(self):
    """
        If one of the L{Response} methods called by
        L{HTTPClientParser.connectionLost} raises an exception, the exception
        is logged and not re-raised.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    transport = StringTransport()
    protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
    protocol.makeConnection(transport)
    response = []
    protocol._responseDeferred.addCallback(response.append)
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n')
    response = response[0]

    def fakeBodyDataFinished(err=None):
        raise ArbitraryException()
    response._bodyDataFinished = fakeBodyDataFinished
    protocol.connectionLost(None)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, ArbitraryException)
    self.flushLoggedErrors(ArbitraryException)