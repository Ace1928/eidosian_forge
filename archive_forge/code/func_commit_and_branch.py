import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def commit_and_branch(self):
    """Commit the current tree, and create a second tree"""
    r1 = self.wt.commit('adding a,b')
    dir2 = self.wt.branch.controldir.sprout('b2', revision_id=r1)
    wt2 = dir2.open_workingtree()
    self.assertEqual([r1], wt2.get_parent_ids())
    self.assertEqual(r1, wt2.branch.last_revision())
    return (wt2, r1)