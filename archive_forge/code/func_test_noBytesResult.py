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
def test_noBytesResult(self):
    """
        When implemented C{render} method does not return bytes an internal
        server error is returned.
        """

    class RiggedRepr:

        def __repr__(self) -> str:
            return 'my>repr'
    result = RiggedRepr()
    no_bytes_resource = resource.Resource()
    no_bytes_resource.render = lambda request: result
    request = self._getReq(no_bytes_resource)
    request.requestReceived(b'GET', b'/newrender', b'HTTP/1.0')
    headers, body = request.transport.written.getvalue().split(b'\r\n\r\n')
    self.assertEqual(request.code, 500)
    expected = ['', '<html>', '  <head><title>500 - Request did not return bytes</title></head>', '  <body>', '    <h1>Request did not return bytes</h1>', '    <p>Request: <pre>&lt;%s&gt;</pre><br />Resource: <pre>&lt;%s&gt;</pre><br />Value: <pre>my&gt;repr</pre></p>' % (reflect.safe_repr(request)[1:-1], reflect.safe_repr(no_bytes_resource)[1:-1]), '  </body>', '</html>', '']
    self.assertEqual('\n'.join(expected).encode('ascii'), body)