import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_merge_to_revision(self):
    """Merge from a branch to a revision that is not the tip."""
    self.create_two_trees_for_merging()
    self.third_rev = self.tree_from.commit('real_tip')
    self.tree_to.merge_from_branch(self.tree_from.branch, to_revision=self.second_rev)
    self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())