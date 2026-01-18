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
def test_getChildChildNotFound_customize(self):
    """
        The resource rendered for child not found requests can be customize
        using a class member.
        """
    base = FilePath(self.mktemp())
    base.setContent(b'')
    markerResponse = b'custom-child-not-found-response'

    class CustomChildNotFoundResource(resource.Resource):

        def render(self, request):
            return markerResponse

    class CustomStaticFile(static.File):
        childNotFound = CustomChildNotFoundResource()
    fileResource = CustomStaticFile(base.path)
    request = DummyRequest([b'no-child.txt'])
    child = fileResource.getChild(b'no-child.txt', request)
    result = child.render(request)
    self.assertEqual(markerResponse, result)