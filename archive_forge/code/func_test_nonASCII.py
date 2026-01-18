import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
def test_nonASCII(self):
    """
        Bytes in fields of the request which are not part of ASCII are escaped
        in the result.
        """
    reactor = Clock()
    reactor.advance(1234567890)
    timestamp = http.datetimeToLogString(reactor.seconds())
    request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
    request.client = IPv4Address('TCP', b'evil x-forwarded-for \x80', 12345)
    request.method = b'POS\x81'
    request.protocol = b'HTTP/1.\x82'
    request.requestHeaders.addRawHeader(b'referer', b'evil \x83')
    request.requestHeaders.addRawHeader(b'user-agent', b'evil \x84')
    line = http.combinedLogFormatter(timestamp, request)
    self.assertEqual('"evil x-forwarded-for \\x80" - - [13/Feb/2009:23:31:30 +0000] "POS\\x81 /dummy HTTP/1.0" 123 - "evil \\x83" "evil \\x84"', line)