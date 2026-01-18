import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_content_file_changes(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])

    def change_file():
        tree.put_file_bytes_non_atomic('file', b'new content')
    self._add_commit_change_check_changed(tree, ('file', 'file'), change_file, expect_fs_hash=True)