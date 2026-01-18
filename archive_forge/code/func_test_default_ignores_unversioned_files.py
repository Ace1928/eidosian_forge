import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_default_ignores_unversioned_files(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree1/a', 'tree1/c', 'tree2/a', 'tree2/b', 'tree2/c'])
    tree1.add(['a', 'c'], ids=[b'a-id', b'c-id'])
    tree2.add(['a', 'c'], ids=[b'a-id', b'c-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected = self.sorted([self.changed_content(tree2, 'a'), self.changed_content(tree2, 'c')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
    self.check_has_changes(True, tree1, tree2)