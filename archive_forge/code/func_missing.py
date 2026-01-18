import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def missing(self, file_id, from_path, to_path, parent_id, kind):
    _, from_basename = os.path.split(from_path)
    _, to_basename = os.path.split(to_path)
    return InventoryTreeChange(file_id, (from_path, to_path), True, (True, True), (parent_id, parent_id), (from_basename, to_basename), (kind, None), (False, False))