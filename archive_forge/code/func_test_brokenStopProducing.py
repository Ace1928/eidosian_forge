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
def test_brokenStopProducing(self):
    """
        If the body producer's C{stopProducing} method raises an exception,
        L{Request.stopWriting} logs it and does not re-raise it.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    producer = StringProducer(3)

    def brokenStopProducing():
        raise ArbitraryException('stopProducing is busted')
    producer.stopProducing = brokenStopProducing
    request = Request(b'GET', b'/', _boringHeaders, producer)
    request.writeTo(self.transport)
    request.stopWriting()
    self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    self.assertIn('log_failure', event)
    f = event['log_failure']
    self.assertIsInstance(f.value, ArbitraryException)