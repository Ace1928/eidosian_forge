import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_reparent_file_changes(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    self._add_commit_reparent_check_changed(tree, 'file', expect_fs_hash=True)