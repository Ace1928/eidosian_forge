import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
def test_explicitRange(self):
    """
        A correct response to a bytes range header request from A to B starts
        with the A'th byte and ends with (including) the B'th byte. The first
        byte of a page is numbered with 0.
        """
    self.request.requestHeaders.addRawHeader(b'range', b'bytes=3-43')
    self.resource.render(self.request)
    written = b''.join(self.request.written)
    self.assertEqual(written, self.payload[3:44])
    self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
    self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 3-43/64')
    self.assertEqual(b'%d' % (len(written),), self.request.responseHeaders.getRawHeaders(b'content-length')[0])