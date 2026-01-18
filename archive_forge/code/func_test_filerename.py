from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filerename(self):
    c = commands.FileRenameCommand(b'foo/bar', b'foo/baz')
    self.assertEqual(b'R foo/bar foo/baz', bytes(c))