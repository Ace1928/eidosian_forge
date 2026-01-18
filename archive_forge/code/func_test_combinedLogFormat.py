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
def test_combinedLogFormat(self):
    """
        The factory's C{log} method writes a I{combined log format} line to the
        factory's log file.
        """
    reactor = Clock()
    reactor.advance(1234567890)
    logPath = self.mktemp()
    factory = self.factory(logPath=logPath, reactor=reactor)
    factory.startFactory()
    try:
        factory.log(DummyRequestForLogTest(factory))
    finally:
        factory.stopFactory()
    self.assertEqual(b'"1.2.3.4" - - [13/Feb/2009:23:31:30 +0000] "GET /dummy HTTP/1.0" 123 - "-" "-"\n', FilePath(logPath).getContent())