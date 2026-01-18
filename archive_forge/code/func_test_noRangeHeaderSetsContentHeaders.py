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
def test_noRangeHeaderSetsContentHeaders(self):
    """
        makeProducer when no Range header is set sets the Content-* headers
        for the response.
        """
    length = 123
    contentType = 'text/plain'
    contentEncoding = 'gzip'
    resource = self.makeResourceWithContent(b'a' * length, type=contentType, encoding=contentEncoding)
    request = DummyRequest([])
    with resource.openForReading() as file:
        resource.makeProducer(request, file)
        self.assertEqual({b'content-type': networkString(contentType), b'content-length': b'%d' % (length,), b'content-encoding': networkString(contentEncoding)}, self.contentHeaders(request))