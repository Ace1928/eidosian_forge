import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def mutable_trees_to_locked_test_trees(self, tree1, tree2):
    """Convert the working trees into test trees.

        Read lock them, and add the unlock to the cleanup.
        """
    tree1, tree2 = self.mutable_trees_to_test_trees(self, tree1, tree2)
    tree1.lock_read()
    self.addCleanup(tree1.unlock)
    tree2.lock_read()
    self.addCleanup(tree2.unlock)
    return (tree1, tree2)