import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_modified_unicode(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    a_id = 'α-id'.encode()
    mod_id = 'ω_mod_id'.encode()
    mod_path = 'α/ω-modified'
    try:
        self.build_tree(['tree1/α/', 'tree1/' + mod_path, 'tree2/α/', 'tree2/' + mod_path])
    except UnicodeError:
        raise tests.TestSkipped('Could not create Unicode files.')
    tree1.add(['α', mod_path], ids=[a_id, mod_id])
    tree2.add(['α', mod_path], ids=[a_id, mod_id])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual([self.changed_content(tree1, mod_path)], self.do_iter_changes(tree1, tree2))
    self.assertEqual([self.changed_content(tree1, mod_path)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
    self.check_has_changes(True, tree1, tree2)