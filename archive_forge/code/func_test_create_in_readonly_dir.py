from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
def test_create_in_readonly_dir(self):
    self.fs.chdir('/')
    self.assertRaises(EnvironmentError, io._open_or_create, self.fs.open, 'foo.bbl', 'wb', {})