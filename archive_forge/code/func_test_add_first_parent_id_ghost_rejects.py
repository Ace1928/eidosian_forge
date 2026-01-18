import os
from io import BytesIO
from ... import errors
from ... import revision as _mod_revision
from ...bzr.inventory import (Inventory, InventoryDirectory, InventoryFile,
from ...bzr.inventorytree import InventoryRevisionTree, InventoryTree
from ...tests import TestNotApplicable
from ...uncommit import uncommit
from .. import features
from ..per_workingtree import TestCaseWithWorkingTree
def test_add_first_parent_id_ghost_rejects(self):
    """Test adding the first parent id - as a ghost"""
    tree = self.make_branch_and_tree('.')
    self.assertRaises(errors.GhostRevisionUnusableHere, tree.add_parent_tree_id, b'first-revision')