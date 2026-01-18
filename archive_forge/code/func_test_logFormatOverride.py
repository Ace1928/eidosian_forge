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
def test_logFormatOverride(self):
    """
        If the factory is initialized with a custom log formatter then that
        formatter is used to generate lines for the log file.
        """

    def notVeryGoodFormatter(timestamp, request):
        return 'this is a bad log format'
    reactor = Clock()
    reactor.advance(1234567890)
    logPath = self.mktemp()
    factory = self.factory(logPath=logPath, logFormatter=notVeryGoodFormatter)
    factory._reactor = reactor
    factory.startFactory()
    try:
        factory.log(DummyRequestForLogTest(factory))
    finally:
        factory.stopFactory()
    self.assertEqual(b'this is a bad log format\n', FilePath(logPath).getContent())