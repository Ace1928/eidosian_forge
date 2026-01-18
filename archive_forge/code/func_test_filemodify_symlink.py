from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filemodify_symlink(self):
    c = commands.FileModifyCommand(b'foo/bar', 40960, None, b'baz')
    self.assertEqual(b'M 120000 inline foo/bar\ndata 3\nbaz', bytes(c))