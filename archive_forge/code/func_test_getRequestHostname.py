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
def test_getRequestHostname(self):
    """
        L{http.Request.getRequestHostname} returns the hostname portion of the
        request, based on the C{Host:} header.
        """
    req = http.Request(DummyChannel(), False)

    def check(header, expectedHost):
        req.requestHeaders.setRawHeaders(b'host', [header])
        self.assertEqual(req.getRequestHostname(), expectedHost)
    check(b'example.com', b'example.com')
    check(b'example.com:8443', b'example.com')
    check(b'192.168.1.1', b'192.168.1.1')
    check(b'192.168.1.1:19289', b'192.168.1.1')
    check(b'[2607:f0d0:1002:51::4]', b'2607:f0d0:1002:51::4')
    check(b'[2607:f0d0:1002:0051:0000:0000:0000:0004]', b'2607:f0d0:1002:0051:0000:0000:0000:0004')
    check(b'[::1]', b'::1')
    check(b'[::1]:8080', b'::1')
    check(b'[2607:f0d0:1002:51::4]:9443', b'2607:f0d0:1002:51::4')