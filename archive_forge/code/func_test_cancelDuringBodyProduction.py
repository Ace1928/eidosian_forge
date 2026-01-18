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
def test_cancelDuringBodyProduction(self):
    """
        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire
        with a L{RequestGenerationFailed} failure containing a
        L{CancelledError} exception if the request was cancelled before a
        C{bodyProducer} with an explicit length has finished producing.
        """
    return self.assertCancelDuringBodyProduction(10)