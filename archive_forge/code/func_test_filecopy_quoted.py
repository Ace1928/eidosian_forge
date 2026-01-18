from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filecopy_quoted(self):
    c = commands.FileCopyCommand(b'foo/b a r', b'foo/b a z')
    self.assertEqual(b'C "foo/b a r" foo/b a z', bytes(c))