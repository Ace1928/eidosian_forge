import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_07_pull(self):
    """Test that pull will handle bits correctly"""
    wt2, r1 = self.commit_and_branch()
    os.remove('b1/a')
    os.remove('b1/b')
    r2 = self.wt.commit('removed')
    wt2.pull(self.wt.branch)
    self.assertEqual([r2], wt2.get_parent_ids())
    self.assertEqual(r2, wt2.branch.last_revision())
    self.check_empty(wt2)
    rev_tree = self.wt.branch.repository.revision_tree(r1)
    self.wt.revert(old_tree=rev_tree, backups=False)
    r3 = self.wt.commit('resurrected')
    self.check_exist(self.wt)
    wt2.pull(self.wt.branch)
    self.assertEqual([r3], wt2.get_parent_ids())
    self.assertEqual(r3, wt2.branch.last_revision())
    self.check_exist(wt2)