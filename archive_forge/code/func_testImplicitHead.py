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
def testImplicitHead(self):
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    req = self._getReq()
    req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
    self.assertEqual(req.code, 200)
    self.assertEqual(-1, req.transport.written.getvalue().find(b'hi hi'))
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    self.assertEquals(event['log_level'], LogLevel.info)