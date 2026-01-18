import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
@skipIf(not http.H2_ENABLED, 'HTTP/2 support not present')
def test_h2CancelsH11Timeout(self):
    """
        When the transport is switched to H2, the HTTPChannel timeouts are
        cancelled.
        """
    clock = Clock()
    a = http._genericHTTPChannelProtocolFactory(b'')
    a.requestFactory = DummyHTTPHandlerProxy
    a.timeOut = 100
    a.callLater = clock.callLater
    b = StringTransport()
    b.negotiatedProtocol = b'h2'
    a.makeConnection(b)
    hamcrest.assert_that(clock.getDelayedCalls(), hamcrest.contains(hamcrest.has_property('cancelled', hamcrest.equal_to(False))))
    h11Timeout = clock.getDelayedCalls()[0]
    a.dataReceived(b'')
    self.assertEqual(a._negotiatedProtocol, b'h2')
    self.assertTrue(h11Timeout.cancelled)
    hamcrest.assert_that(clock.getDelayedCalls(), hamcrest.contains(hamcrest.has_property('cancelled', hamcrest.equal_to(False))))