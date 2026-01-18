import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_rename_link_changes(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    tree = self.make_branch_and_tree('.')
    os.symlink('target', 'link')
    self._add_commit_renamed_check_changed(tree, 'link')