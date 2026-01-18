import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_non_relative(self):
    result = urlutils.rebase_url('file://foo', 'file://foo', 'file://foo/bar')
    self.assertEqual('file://foo', result)
    result = urlutils.rebase_url('/foo', 'file://foo', 'file://foo/bar')
    self.assertEqual('/foo', result)