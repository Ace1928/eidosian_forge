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
def test_chunkedEncoding(self):
    """
        If a request uses the I{chunked} transfer encoding, the request body is
        decoded accordingly before it is made available on the request.
        """
    httpRequest = b'GET / HTTP/1.0\nContent-Type: text/plain\nTransfer-Encoding: chunked\n\n6\nHello,\n14\n spam,eggs spam spam\n0\n\n'
    path = []
    method = []
    content = []
    decoder = []
    testcase = self

    class MyRequest(http.Request):

        def process(self):
            content.append(self.content)
            content.append(self.content.read())
            self.content = BytesIO()
            method.append(self.method)
            path.append(self.path)
            decoder.append(self.channel._transferDecoder)
            testcase.didRequest = True
            self.finish()
    self.runRequest(httpRequest, MyRequest)
    self.addCleanup(content[0].close)
    assertIsFilesystemTemporary(self, content[0])
    self.assertEqual(content[1], b'Hello, spam,eggs spam spam')
    self.assertEqual(method, [b'GET'])
    self.assertEqual(path, [b'/'])
    self.assertEqual(decoder, [None])