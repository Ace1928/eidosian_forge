import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_merge_base(self):
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'text-a')])
    tree_a.add('file')
    rev1 = tree_a.commit('added file')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    os.unlink('tree_a/file')
    tree_a.commit('deleted file')
    self.build_tree_contents([('tree_b/file', b'text-b')])
    tree_b.commit('changed file')
    self.assertRaises(PointlessMerge, tree_a.merge_from_branch, tree_b.branch, from_revision=tree_b.branch.last_revision())
    tree_a.merge_from_branch(tree_b.branch, from_revision=rev1)
    tree_a.lock_read()
    self.addCleanup(tree_a.unlock)
    changes = list(tree_a.iter_changes(tree_a.basis_tree()))
    self.assertEqual(1, len(changes), changes)