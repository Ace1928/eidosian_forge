from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
def test_locate(self):
    file = io._open_existing(self.fs.open, 'unsrt.bst', 'rb', locate=self.fs.locate)
    self.assertEqual(file.name, '/usr/share/texmf/bibtex/bst/unsrt.bst')
    self.assertEqual(file.mode, 'rb')