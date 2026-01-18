from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_filemodify_treeref(self):
    c = commands.FileModifyCommand(b'tree-info', 57344, b'revision-id-info', None)
    self.assertEqual(b'M 160000 revision-id-info tree-info', bytes(c))