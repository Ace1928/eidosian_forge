import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_merge_empty(self):
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'text-a')])
    tree_a.add('file')
    tree_a.commit('added file')
    tree_b = self.make_branch_and_tree('treeb')
    self.assertRaises(errors.NoCommits, tree_a.merge_from_branch, tree_b.branch)
    tree_b.merge_from_branch(tree_a.branch)