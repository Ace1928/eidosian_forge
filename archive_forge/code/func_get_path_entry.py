import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
@staticmethod
def get_path_entry(tree, path):
    iterator = tree.iter_entries_by_dir(specific_files=[path])
    try:
        return next(iterator)[1]
    except StopIteration:
        raise KeyError(path)