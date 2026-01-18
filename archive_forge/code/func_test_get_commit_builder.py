import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_get_commit_builder(self):
    branch = self.make_branch('.')
    branch.repository.lock_write()
    builder = branch.repository.get_commit_builder(branch, [], branch.get_config_stack())
    self.assertIsInstance(builder, repository.CommitBuilder)
    self.assertTrue(builder.random_revid)
    branch.repository.commit_write_group()
    branch.repository.unlock()