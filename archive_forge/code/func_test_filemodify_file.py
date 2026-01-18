from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filemodify_file(self):
    c = commands.FileModifyCommand(b'foo/bar', 33188, b':23', None)
    self.assertEqual(b'M 644 :23 foo/bar', bytes(c))