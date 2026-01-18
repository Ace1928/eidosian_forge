import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_missing_in_target(self):
    """Test with the target files versioned but absent from disk."""
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_abc_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content(tree2)
    os.unlink('2/a')
    shutil.rmtree('2/b')
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    self.not_applicable_if_missing_in('a', tree2)
    self.not_applicable_if_missing_in('b', tree2)
    expected = self.sorted([self.missing(b'a-id', 'a', 'a', b'root-id', 'file'), self.missing(b'b-id', 'b', 'b', b'root-id', 'directory'), self.missing(b'c-id', 'b/c', 'b/c', b'b-id', 'file')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))