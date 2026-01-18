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
def assertTransitionFromBasisToShape(self, basis_shape, basis_revid, new_shape, new_revid, extra_parent=None, set_current_inventory=True):
    basis_shape.revision_id = basis_revid
    new_shape.revision_id = new_revid
    delta = self.make_inv_delta(basis_shape, new_shape)
    tree = self.make_branch_and_tree('tree')
    if basis_revid is not None:
        self.fake_up_revision(tree, basis_revid, basis_shape)
        parents = [basis_revid]
        if extra_parent is not None:
            parents.append(extra_parent)
        tree.set_parent_ids(parents)
    self.fake_up_revision(tree, new_revid, new_shape)
    if set_current_inventory:
        tree._write_inventory(new_shape)
    self.assertDeltaApplicationResultsInExpectedBasis(tree, new_revid, delta, new_shape)
    tree._validate()
    if tree.user_url != tree.branch.user_url:
        tree.branch.controldir.root_transport.delete_tree('.')
    tree.controldir.root_transport.delete_tree('.')