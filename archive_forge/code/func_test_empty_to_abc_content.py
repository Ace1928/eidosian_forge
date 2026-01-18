import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_empty_to_abc_content(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_no_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content(tree2)
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected_results = self.sorted([self.added(tree2, ''), self.added(tree2, 'a'), self.added(tree2, 'b'), self.added(tree2, 'b/c'), self.deleted(tree1, '')])
    self.assertEqual(expected_results, self.do_iter_changes(tree1, tree2))
    self.check_has_changes(True, tree1, tree2)