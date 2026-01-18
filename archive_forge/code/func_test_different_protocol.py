import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_different_protocol(self):
    e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar', 'ftp://bar')
    self.assertEqual(str(e), "URLs differ by more than path: 'http://bar' and 'ftp://bar'")