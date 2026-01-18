import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_revision_after_merge_dir_changes(self):
    tree1 = self.make_branch_and_tree('t1')
    if not tree1.has_versioned_directories():
        raise tests.TestNotApplicable('Format does not support versioned directories')
    self.build_tree(['t1/dir/'])
    self._commit_sprout_rename_merge(tree1, 'dir')