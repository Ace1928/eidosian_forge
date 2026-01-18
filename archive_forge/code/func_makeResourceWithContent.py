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
def makeResourceWithContent(self, content, type=None, encoding=None):
    """
        Make a L{static.File} resource that has C{content} for its content.

        @param content: The L{bytes} to use as the contents of the resource.
        @param type: Optional value for the content type of the resource.
        """
    fileName = FilePath(self.mktemp())
    fileName.setContent(content)
    resource = static.File(fileName._asBytesPath())
    resource.encoding = encoding
    resource.type = type
    return resource