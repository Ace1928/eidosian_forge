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
def test_multipleRequestsInDifferentSegments(self) -> None:
    """
        Twisted MUST NOT respond to a second HTTP/1.1 request while the first
        is still pending, even if the second request is received in a separate
        TCP package.
        """
    qr = QueueResource()
    site = Site(qr)
    proto = site.buildProtocol(None)
    serverTransport = StringTransport()
    proto.makeConnection(serverTransport)
    raw_data = b'GET /first HTTP/1.1\r\nHost: a\r\n\r\nGET /second HTTP/1.1\r\nHost: a\r\n\r\n'
    for chunk in iterbytes(raw_data):
        proto.dataReceived(chunk)
    self.assertEqual(len(qr.dispatchedRequests), 1)
    qr.dispatchedRequests[0].finish()
    self.assertEqual(len(qr.dispatchedRequests), 2)