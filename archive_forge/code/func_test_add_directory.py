import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_add_directory(self):
    """Test that adding a directory will strip any trailing slash"""
    ignores._set_user_ignores([])
    in_patterns = ['foo/', 'bar/', 'baz\\']
    added = ignores.add_unique_user_ignores(in_patterns)
    out_patterns = [x.rstrip('/\\') for x in in_patterns]
    self.assertEqual(out_patterns, added)
    self.assertEqual(set(out_patterns), ignores.get_user_ignores())