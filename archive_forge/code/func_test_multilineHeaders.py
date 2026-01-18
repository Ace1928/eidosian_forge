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
def test_multilineHeaders(self):
    """
        L{HTTPClient} parses multiline headers by buffering header lines until
        an empty line or a line that does not start with whitespace hits
        lineReceived, confirming that the header has been received completely.
        """
    c = ClientDriver()
    c.handleHeader = self.ourHandleHeader
    c.handleEndHeaders = self.ourHandleEndHeaders
    c.lineReceived(b'HTTP/1.0 201')
    c.lineReceived(b'X-Multiline: line-0')
    self.assertFalse(self.handleHeaderCalled)
    c.lineReceived(b'\tline-1')
    c.lineReceived(b'X-Multiline2: line-2')
    self.assertTrue(self.handleHeaderCalled)
    c.lineReceived(b' line-3')
    c.lineReceived(b'Content-Length: 10')
    c.lineReceived(b'')
    self.assertTrue(self.handleEndHeadersCalled)
    self.assertEqual(c.version, b'HTTP/1.0')
    self.assertEqual(c.status, b'201')
    self.assertEqual(c.length, 10)