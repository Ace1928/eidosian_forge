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
def test_clientAddrIPv6(self):
    """
        A request from an IPv6 client is logged with that IP address.
        """
    reactor = Clock()
    reactor.advance(1234567890)
    timestamp = http.datetimeToLogString(reactor.seconds())
    request = DummyRequestForLogTest(http.HTTPFactory(reactor=reactor))
    request.client = IPv6Address('TCP', b'::1', 12345)
    line = http.combinedLogFormatter(timestamp, request)
    self.assertEqual('"::1" - - [13/Feb/2009:23:31:30 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"', line)