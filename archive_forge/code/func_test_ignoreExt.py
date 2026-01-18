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
def test_ignoreExt(self):
    """
        The list of ignored extensions can be set by passing a value to
        L{File.__init__} or by calling L{File.ignoreExt} later.
        """
    file = static.File(b'.')
    self.assertEqual(file.ignoredExts, [])
    file.ignoreExt('.foo')
    file.ignoreExt('.bar')
    self.assertEqual(file.ignoredExts, ['.foo', '.bar'])
    file = static.File(b'.', ignoredExts=('.bar', '.baz'))
    self.assertEqual(file.ignoredExts, ['.bar', '.baz'])