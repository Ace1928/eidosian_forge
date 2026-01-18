from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filecopy(self):
    c = commands.FileCopyCommand(b'foo/bar', b'foo/baz')
    self.assertEqual(b'C foo/bar foo/baz', bytes(c))