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
def test_no_parents_full_tree(self):
    """Test doing a regular initial commit with files and dirs."""
    basis_shape = Inventory(root_id=None)
    revid = b'new-parent'
    new_shape = Inventory(root_id=None)
    self.add_dir(new_shape, revid, b'root-id', None, '')
    self.add_link(new_shape, revid, b'link-id', b'root-id', 'link', 'target')
    self.add_file(new_shape, revid, b'file-id', b'root-id', 'file', b'1' * 32, 12)
    self.add_dir(new_shape, revid, b'dir-id', b'root-id', 'dir')
    self.add_file(new_shape, revid, b'subfile-id', b'dir-id', 'subfile', b'2' * 32, 24)
    self.assertTransitionFromBasisToShape(basis_shape, None, new_shape, revid)