import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_02_stays_executable(self):
    """reopen the tree and ensure it stuck."""
    self.wt = self.wt.controldir.open_workingtree()
    self.check_exist(self.wt)