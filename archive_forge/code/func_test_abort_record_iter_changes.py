import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_abort_record_iter_changes(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        builder = tree.branch.get_commit_builder([])
        try:
            basis = tree.basis_tree()
            last_rev = tree.last_revision()
            changes = tree.iter_changes(basis)
            list(builder.record_iter_changes(tree, last_rev, changes))
            builder.finish_inventory()
        finally:
            builder.abort()