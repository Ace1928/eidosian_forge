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
def test_ignoredExtFalse(self):
    """
        Passing C{1} as the value to L{File}'s C{ignoredExts} argument
        issues a warning and sets the ignored extensions to the empty
        list.
        """
    with warnings.catch_warnings(record=True) as caughtWarnings:
        file = static.File(self.mktemp(), ignoredExts=0)
        self.assertEqual(file.ignoredExts, [])
    self.assertEqual(len(caughtWarnings), 1)