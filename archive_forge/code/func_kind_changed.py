import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def kind_changed(self, from_tree, to_tree, from_path, to_path):
    old_entry = self.get_path_entry(from_tree, from_path)
    new_entry = self.get_path_entry(to_tree, to_path)
    return InventoryTreeChange(new_entry.file_id, (from_path, to_path), True, (True, True), (old_entry.parent_id, new_entry.parent_id), (old_entry.name, new_entry.name), (old_entry.kind, new_entry.kind), (old_entry.executable, new_entry.executable))