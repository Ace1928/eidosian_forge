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
def test_formPOSTRequest(self):
    """
        The request body of a I{POST} request with a I{Content-Type} header
        of I{application/x-www-form-urlencoded} is parsed according to that
        content type and made available in the C{args} attribute of the
        request object.  The original bytes of the request may still be read
        from the C{content} attribute.
        """
    query = 'key=value&multiple=two+words&multiple=more%20words&empty='
    httpRequest = networkString('POST / HTTP/1.0\nContent-Length: %d\nContent-Type: application/x-www-form-urlencoded\n\n%s' % (len(query), query))
    method = []
    args = []
    content = []
    testcase = self

    class MyRequest(http.Request):

        def process(self):
            method.append(self.method)
            args.extend([self.args[b'key'], self.args[b'empty'], self.args[b'multiple']])
            content.append(self.content.read())
            testcase.didRequest = True
            self.finish()
    self.runRequest(httpRequest, MyRequest)
    self.assertEqual(method, [b'POST'])
    self.assertEqual(args, [[b'value'], [b''], [b'two words', b'more words']])
    self.assertEqual(content, [networkString(query)])