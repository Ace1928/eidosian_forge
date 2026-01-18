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
def test_multipleEncodingLines(self):
    """
        If there are several I{Content-Encoding} headers,
        L{server.GzipEncoderFactory} normalizes it and appends gzip to the
        field value.
        """
    request = server.Request(self.channel, False)
    request.gotLength(0)
    request.requestHeaders.setRawHeaders(b'Accept-Encoding', [b'deflate', b'gzip'])
    request.responseHeaders.setRawHeaders(b'Content-Encoding', [b'foo', b'bar'])
    request.requestReceived(b'GET', b'/foo', b'HTTP/1.0')
    data = self.channel.transport.written.getvalue()
    self.assertNotIn(b'Content-Length', data)
    self.assertIn(b'Content-Encoding: foo,bar,gzip\r\n', data)
    body = data[data.find(b'\r\n\r\n') + 4:]
    self.assertEqual(b'Some data', zlib.decompress(body, 16 + zlib.MAX_WBITS))