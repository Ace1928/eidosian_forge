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
def test_headersMultiline(self):
    """
        Line folded headers are handled by L{HTTPChannel} by replacing each
        fold with a single space by the time they are made available to the
        L{Request}. Any leading whitespace in the folded lines of the header
        value is replaced with a single space, per:

            A server that receives an obs-fold in a request message ... MUST
            ... replace each received obs-fold with one or more SP octets prior
            to interpreting the field value or forwarding the message
            downstream.

        See RFC 7230 section 3.2.4.
        """
    processed = []

    class MyRequest(http.Request):

        def process(self):
            processed.append(self)
            self.finish()
    requestLines = [b'GET / HTTP/1.0', b'nospace: ', b' nospace\t', b'space:space', b' space', b'spaces: spaces', b'  spaces', b'   spaces', b'tab: t', b'\ta', b'\tb', b'', b'']
    self.runRequest(b'\n'.join(requestLines), MyRequest, 0)
    [request] = processed
    self.assertEqual(request.requestHeaders.getRawHeaders(b'nospace'), [b'nospace'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'space'), [b'space space'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'spaces'), [b'spaces spaces spaces'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'tab'), [b't a b'])