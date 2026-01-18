import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_determine_relative_path(self):
    self.assertEqual('../../baz/bar', urlutils.determine_relative_path('/qux/quxx', '/baz/bar'))
    self.assertEqual('..', urlutils.determine_relative_path('/bar/baz', '/bar'))
    self.assertEqual('baz', urlutils.determine_relative_path('/bar', '/bar/baz'))
    self.assertEqual('.', urlutils.determine_relative_path('/bar', '/bar'))