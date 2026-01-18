from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
def test_create_in_fallback_dir(self):
    self.fs.chdir('/')
    file = io._open_or_create(self.fs.open, 'foo.bbl', 'wb', {'TEXMFOUTPUT': '/home/test'})
    self.assertEqual(file.name, '/home/test/foo.bbl')
    self.assertEqual(file.mode, 'wb')