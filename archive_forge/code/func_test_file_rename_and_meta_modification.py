import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_file_rename_and_meta_modification(self):
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_abc_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content_6(tree2)
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    root_id = tree1.path2id('')
    self.assertEqual([(b'c-id', ('b/c', 'e'), False, (True, True), (b'b-id', root_id), ('c', 'e'), ('file', 'file'), (False, True), False)], self.do_iter_changes(tree1, tree2))