import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def changed_content(self, tree, path):
    entry = self.get_path_entry(tree, path)
    return InventoryTreeChange(entry.file_id, (path, path), True, (True, True), (entry.parent_id, entry.parent_id), (entry.name, entry.name), (entry.kind, entry.kind), (entry.executable, entry.executable))