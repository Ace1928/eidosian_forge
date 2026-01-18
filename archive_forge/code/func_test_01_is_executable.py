import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_01_is_executable(self):
    """Make sure that the tree was created and has the executable bit set"""
    self.check_exist(self.wt)