from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filecopy_path_checking(self):
    self.assertRaises(ValueError, commands.FileCopyCommand, b'', b'foo')
    self.assertRaises(ValueError, commands.FileCopyCommand, None, b'foo')
    self.assertRaises(ValueError, commands.FileCopyCommand, b'foo', b'')
    self.assertRaises(ValueError, commands.FileCopyCommand, b'foo', None)