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
def test_headersTooBigPerRequest(self):
    """
        Enforces total size of headers per individual request and counter
        is reset at the end of each request.
        """

    class SimpleRequest(http.Request):

        def process(self):
            self.finish()
    channel = http.HTTPChannel()
    channel.totalHeadersSize = 60
    channel.requestFactory = SimpleRequest
    httpRequest = b'GET / HTTP/1.1\nSome-Header: total-less-than-60\n\nGET / HTTP/1.1\nSome-Header: less-than-60\n\n'
    channel = self.runRequest(httpRequest=httpRequest, channel=channel, success=False)
    self.assertEqual(channel.transport.value(), b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nHTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n')