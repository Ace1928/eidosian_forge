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
@skipIf(not platform._supportsSymlinks(), 'No symlink support')
def test_brokenSymlink(self):
    """
        If on the file in the listing points to a broken symlink, it should not
        be returned by L{static.DirectoryLister._getFilesAndDirectories}.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    file1 = path.child('file1')
    file1.setContent(b'file1')
    file1.linkTo(path.child('file2'))
    file1.remove()
    lister = static.DirectoryLister(path.path)
    directory = os.listdir(path.path)
    directory.sort()
    dirs, files = lister._getFilesAndDirectories(directory)
    self.assertEqual(dirs, [])
    self.assertEqual(files, [])