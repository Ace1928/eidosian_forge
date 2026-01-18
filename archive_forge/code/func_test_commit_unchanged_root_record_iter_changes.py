import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_commit_unchanged_root_record_iter_changes(self):
    tree = self.make_branch_and_tree('.')
    old_revision_id = tree.commit('oldrev')
    tree.lock_write()
    builder = tree.branch.get_commit_builder([old_revision_id])
    try:
        list(builder.record_iter_changes(tree, old_revision_id, []))
        self.assertFalse(builder.any_changes())
        builder.finish_inventory()
        builder.commit('rev')
        builder_tree = builder.revision_tree()
        new_root_revision = builder_tree.get_file_revision('')
        if tree.branch.repository.supports_rich_root():
            self.assertEqual(old_revision_id, new_root_revision)
        else:
            self.assertNotEqual(old_revision_id, new_root_revision)
    finally:
        tree.unlock()