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
def test_invalidRangeRequest(self):
    """
        An incorrect range request (RFC 2616 defines a correct range request as
        a Bytes-Unit followed by a '=' character followed by a specific range.
        Only 'bytes' is defined) results in the range header value being logged
        and a normal 200 response being sent.
        """
    range = b'foobar=0-43'
    self.request.requestHeaders.addRawHeader(b'range', range)
    self.resource.render(self.request)
    expected = f'Ignoring malformed Range header {range.decode()!r}'
    self._assertLogged(expected)
    self.assertEqual(b''.join(self.request.written), self.payload)
    self.assertEqual(self.request.responseCode, http.OK)
    self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'%d' % (len(self.payload),))