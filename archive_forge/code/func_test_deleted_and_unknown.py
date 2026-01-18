import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_deleted_and_unknown(self):
    """Test a file marked removed, but still present on disk."""
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree2/a', b'a contents\n'), ('tree2/b', b'b contents\n'), ('tree2/c', b'c contents\n')])
    tree1.add(['a', 'b', 'c'], ids=[b'a-id', b'b-id', b'c-id'])
    tree2.add(['a', 'c'], ids=[b'a-id', b'c-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    expected = self.sorted([self.deleted(tree1, 'b'), self.unversioned(tree2, 'b')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))
    expected = self.sorted([self.deleted(tree1, 'b')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=False))