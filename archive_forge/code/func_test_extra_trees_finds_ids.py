import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_extra_trees_finds_ids(self):
    """Ask for a delta between two trees with a path present in a third."""
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_abc_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content_3(tree2)
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    d = self.intertree_class(tree1, tree2).compare(specific_files=['b'])
    tree3 = self.make_branch_and_tree('3')
    tree3 = self.get_tree_no_parents_abc_content_6(tree3)
    tree3.lock_read()
    self.addCleanup(tree3.unlock)
    d = self.intertree_class(tree1, tree2).compare(specific_files=['e'])
    self.assertEqual([], d.modified)
    d = self.intertree_class(tree1, tree2).compare(specific_files=['e'], extra_trees=[tree3])
    self.assertEqual([], d.added)
    self.assertEqual([('b/c', 'file', False, True)], [(c.path[1], c.kind[1], c.changed_content, c.meta_modified()) for c in d.modified])
    self.assertEqual([], d.removed)
    self.assertEqual([], d.renamed)
    self.assertEqual([], d.unchanged)