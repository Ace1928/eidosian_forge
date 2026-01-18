import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_different_ports(self):
    e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar:80', 'http://bar:81')
    self.assertEqual(str(e), "URLs differ by more than path: 'http://bar:80' and 'http://bar:81'")