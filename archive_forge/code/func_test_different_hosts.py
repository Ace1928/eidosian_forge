import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_different_hosts(self):
    e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar', 'http://baz')
    self.assertEqual(str(e), "URLs differ by more than path: 'http://bar' and 'http://baz'")