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
def test_OPTIONSStar(self):
    """
        L{Request} handles OPTIONS * requests by doing a fast-path return of
        200 OK.
        """
    d = DummyChannel()
    request = server.Request(d, 1)
    request.setHost(b'example.com', 80)
    request.gotLength(0)
    request.requestReceived(b'OPTIONS', b'*', b'HTTP/1.1')
    response = d.transport.written.getvalue()
    self.assertTrue(response.startswith(b'HTTP/1.1 200 OK'))
    self.assertIn(b'Content-Length: 0\r\n', response)