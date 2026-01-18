import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_missing_and_renamed(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree1/file'])
    tree1.add(['file'], ids=[b'file-id'])
    self.build_tree(['tree2/directory/'])
    tree2.add(['directory'], ids=[b'file-id'])
    os.rmdir('tree2/directory')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_missing_in('directory', tree2)
    root_id = tree1.path2id('')
    expected = self.sorted([self.missing(b'file-id', 'file', 'directory', root_id, 'file')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))