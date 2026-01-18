import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_commit_without_root_errors(self):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        builder = tree.branch.get_commit_builder([])

        def do_commit():
            try:
                list(builder.record_iter_changes(tree, tree.last_revision(), []))
                builder.finish_inventory()
            except:
                builder.abort()
                raise
            else:
                builder.commit('msg')
        self.assertRaises(errors.RootMissing, do_commit)