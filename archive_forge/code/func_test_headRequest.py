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
def test_headRequest(self):
    """
        L{static.File.render} returns an empty response body for I{HEAD}
        requests.
        """
    path = FilePath(self.mktemp())
    path.setContent(b'foo')
    file = static.File(path.path)
    request = DummyRequest([b''])
    request.method = b'HEAD'
    d = _render(file, request)

    def cbRendered(ignored):
        self.assertEqual(b''.join(request.written), b'')
    d.addCallback(cbRendered)
    return d