import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_reference(self):
    tree, sub_tree = self.make_nested_trees()
    with tree.lock_write():
        if tree.supports_setting_file_ids():
            sub_tree_root_id = sub_tree.path2id('')
            self.assertEqual(tree.path2id('sub-tree'), sub_tree_root_id)
        self.assertEqual(tree.kind('sub-tree'), 'tree-reference')
        tree.commit('commit reference')
        basis = tree.basis_tree()
        with basis.lock_read():
            sub_tree = tree.get_nested_tree('sub-tree')
            self.assertEqual(sub_tree.last_revision(), tree.get_reference_revision('sub-tree'))
    self.assertEqual(['sub-tree'], list(tree.iter_references()))