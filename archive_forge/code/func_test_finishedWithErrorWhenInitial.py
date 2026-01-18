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
def test_finishedWithErrorWhenInitial(self):
    """
        The L{Failure} passed to L{Response._bodyDataFinished} when the response
        is in the I{initial} state is passed to the C{connectionLost} method of
        the L{IProtocol} provider passed to the L{Response}'s C{deliverBody}
        method.
        """
    transport = StringTransport()
    response = justTransportResponse(transport)
    self.assertEqual(response._state, 'INITIAL')
    response._bodyDataFinished(Failure(ArbitraryException()))
    protocol = AccumulatingProtocol()
    response.deliverBody(protocol)
    protocol.closedReason.trap(ArbitraryException)