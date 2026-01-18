import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def deleted(self, tree, path):
    entry = self.get_path_entry(tree, path)
    return InventoryTreeChange(entry.file_id, (path, None), True, (True, False), (entry.parent_id, None), (entry.name, None), (entry.kind, None), (entry.executable, None))