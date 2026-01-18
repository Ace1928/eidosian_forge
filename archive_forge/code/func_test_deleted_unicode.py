import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_deleted_unicode(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    a_id = 'α-id'.encode()
    deleted_id = 'ω_deleted_id'.encode()
    deleted_path = 'α/ω-deleted'
    try:
        self.build_tree(['tree1/α/', 'tree1/α/ω-deleted', 'tree2/α/'])
    except UnicodeError:
        raise tests.TestSkipped('Could not create Unicode files.')
    tree1.add(['α', 'α/ω-deleted'], ids=[a_id, deleted_id])
    tree2.add(['α'], ids=[a_id])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual([self.deleted(tree1, deleted_path)], self.do_iter_changes(tree1, tree2))
    self.assertEqual([self.deleted(tree1, deleted_path)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
    self.check_has_changes(True, tree1, tree2)