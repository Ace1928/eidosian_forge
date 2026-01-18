import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_04_after_removed(self):
    """Make sure reverting removed files brings them back correctly"""
    r1 = self.wt.commit('adding a,b')
    os.remove('b1/a')
    os.remove('b1/b')
    self.check_empty(self.wt, ignore_inv=True)
    rev_tree = self.wt.branch.repository.revision_tree(r1)
    self.wt.revert(['a', 'b'], rev_tree, backups=False)
    self.check_exist(self.wt)