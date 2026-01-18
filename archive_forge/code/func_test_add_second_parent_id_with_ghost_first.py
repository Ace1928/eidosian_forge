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
def test_add_second_parent_id_with_ghost_first(self):
    """Test adding the second parent when the first is a ghost."""
    tree = self.make_branch_and_tree('.')
    try:
        tree.add_parent_tree_id(b'first-revision', allow_leftmost_as_ghost=True)
    except errors.GhostRevisionUnusableHere:
        self.assertFalse(tree._format.supports_leftmost_parent_id_as_ghost)
    else:
        tree.add_parent_tree_id(b'second')
        self.assertConsistentParents([b'first-revision', b'second'], tree)