import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def do_iter_changes(self, tree1, tree2, **extra_args):
    """Helper to run iter_changes from tree1 to tree2.

        :param tree1, tree2:  The source and target trees. These will be locked
            automatically.
        :param **extra_args: Extra args to pass to iter_changes. This is not
            inspected by this test helper.
        """
    with tree1.lock_read(), tree2.lock_read():
        return self.sorted(self.intertree_class(tree1, tree2).iter_changes(**extra_args))