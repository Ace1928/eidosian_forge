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
def test_multipartProcessingFailure(self):
    """
        When the multipart processing fails the client gets a 400 Bad Request.
        """
    req = b'POST / HTTP/1.0\nContent-Type: multipart/form-data; boundary=\xe2\x98\x83\nContent-Length: 103\n\n--\xe2\x98\x83\nContent-Type: text/plain\nContent-Length: 999999999999999999999999999999999999999999999999999999999999999\nContent-Transfer-Encoding: quoted-printable\n\nabasdfg\n--\xe2\x98\x83--\n'
    channel = self.runRequest(req, http.Request, success=False)
    self.assertEqual(channel.transport.value(), b'HTTP/1.1 400 Bad Request\r\n\r\n')