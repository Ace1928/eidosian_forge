import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_get_containing_nested_tree(self):
    tree, sub_tree = self.make_nested_trees()
    self.build_tree_contents([('tree/sub-tree/foo', 'bar')])
    sub_tree.add('foo')
    sub_tree.commit('rev1')
    with tree.lock_read():
        sub_tree2, subpath = tree.get_containing_nested_tree('sub-tree/foo')
        self.assertEqual(sub_tree.basedir, sub_tree2.basedir)
        self.assertEqual(subpath, 'foo')
        self.assertEqual((None, None), tree.get_containing_nested_tree('not-subtree/bar'))