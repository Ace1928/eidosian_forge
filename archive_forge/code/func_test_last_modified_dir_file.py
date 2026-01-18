import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_last_modified_dir_file(self):
    if not self.repository_format.supports_versioned_directories:
        raise tests.TestNotApplicable('format does not support versioned directories')
    try:
        self._check_kind_change(self.make_dir, self.make_file, expect_fs_hash=True)
    except errors.UnsupportedKindChange:
        raise tests.TestSkipped('tree does not support changing entry kind from directory to file')