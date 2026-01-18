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
def test_errorMessageOnConnectionLostBeforeGenerationFailedDoesNotConfuse(self):
    """
        If the request passed to L{HTTP11ClientProtocol} finished generation
        with an error after the L{HTTP11ClientProtocol}'s connection has been
        lost, an error is logged that gives a non-confusing hint to user on what
        went wrong.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

    def check(ignore):
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertIn('log_failure', event)
        self.assertEqual(event['log_format'], 'Error writing request, but not in valid state to finalize request: {state}')
        self.assertEqual(event['state'], 'CONNECTION_LOST')
    return self.test_connectionLostDuringRequestGeneration('errback').addCallback(check)