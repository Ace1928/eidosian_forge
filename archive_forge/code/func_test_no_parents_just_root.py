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
def test_no_parents_just_root(self):
    """Test doing an empty commit - no parent, set a root only."""
    basis_shape = Inventory(root_id=None)
    new_shape = Inventory()
    self.assertTransitionFromBasisToShape(basis_shape, None, new_shape, b'new_parent')