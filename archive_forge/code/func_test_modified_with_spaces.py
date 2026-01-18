import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_modified_with_spaces(self):
    """Test that 'modified' command reports modified files with spaces in their names quoted"""
    self._test_modified('a filename with spaces', '"a filename with spaces"')