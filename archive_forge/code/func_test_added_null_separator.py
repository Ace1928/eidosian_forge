import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_added_null_separator(self):
    """Test that added uses its null operator properly"""
    self._test_added('a', 'a\x00', null=True)