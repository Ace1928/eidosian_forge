import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_added(self):
    """Test that 'added' command reports added files"""
    self._test_added('a', 'a\n')