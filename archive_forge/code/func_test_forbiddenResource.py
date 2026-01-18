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
@skipIf(platform.isWindows(), 'Cannot remove read permission on Windows')
def test_forbiddenResource(self):
    """
        If the file in the filesystem which would satisfy a request cannot be
        read, L{File.render} sets the HTTP response code to I{FORBIDDEN}.
        """
    base = FilePath(self.mktemp())
    base.setContent(b'')
    self.addCleanup(base.chmod, 448)
    base.chmod(0)
    file = static.File(base.path)
    request = DummyRequest([b''])
    d = self._render(file, request)

    def cbRendered(ignored):
        self.assertEqual(request.responseCode, 403)
    d.addCallback(cbRendered)
    return d