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
def test_mimeTypeAndEncodings(self):
    """
        L{static.DirectoryLister} is able to detect mimetype and encoding of
        listed files.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    path.child('file1.txt').setContent(b'file1')
    path.child('file2.py').setContent(b'python')
    path.child('file3.conf.gz').setContent(b'conf compressed')
    path.child('file4.diff.bz2').setContent(b'diff compressed')
    directory = os.listdir(path.path)
    directory.sort()
    contentTypes = {'.txt': 'text/plain', '.py': 'text/python', '.conf': 'text/configuration', '.diff': 'text/diff'}
    lister = static.DirectoryLister(path.path, contentTypes=contentTypes)
    dirs, files = lister._getFilesAndDirectories(directory)
    self.assertEqual(dirs, [])
    self.assertEqual(files, [{'encoding': '', 'href': 'file1.txt', 'size': '5B', 'text': 'file1.txt', 'type': '[text/plain]'}, {'encoding': '', 'href': 'file2.py', 'size': '6B', 'text': 'file2.py', 'type': '[text/python]'}, {'encoding': '[gzip]', 'href': 'file3.conf.gz', 'size': '15B', 'text': 'file3.conf.gz', 'type': '[text/configuration]'}, {'encoding': '[bzip2]', 'href': 'file4.diff.bz2', 'size': '15B', 'text': 'file4.diff.bz2', 'type': '[text/diff]'}])