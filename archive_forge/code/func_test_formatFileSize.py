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
def test_formatFileSize(self):
    """
        L{static.formatFileSize} format an amount of bytes into a more readable
        format.
        """
    self.assertEqual(static.formatFileSize(0), '0B')
    self.assertEqual(static.formatFileSize(123), '123B')
    self.assertEqual(static.formatFileSize(4567), '4K')
    self.assertEqual(static.formatFileSize(8900000), '8M')
    self.assertEqual(static.formatFileSize(1234000000), '1G')
    self.assertEqual(static.formatFileSize(1234567890000), '1149G')