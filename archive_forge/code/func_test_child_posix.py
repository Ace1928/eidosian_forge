import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_child_posix(self):
    self._with_posix_paths()
    self.assertEqual('b', urlutils.file_relpath('file:///a', 'file:///a/b'))
    self.assertEqual('b', urlutils.file_relpath('file:///a/', 'file:///a/b'))
    self.assertEqual('b/c', urlutils.file_relpath('file:///a', 'file:///a/b/c'))