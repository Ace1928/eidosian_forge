from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def populate_repository(self, repo):
    inv = self.make_one_file_inventory(repo, b'parent-1', [])
    self.add_revision(repo, b'parent-1', inv, [])
    inv = self.make_one_file_inventory(repo, b'parent-2', [])
    self.add_revision(repo, b'parent-2', inv, [])
    inv = self.make_one_file_inventory(repo, b'broken-revision-1-2', [b'parent-2', b'parent-1'])
    self.add_revision(repo, b'broken-revision-1-2', inv, [b'parent-1', b'parent-2'])
    inv = self.make_one_file_inventory(repo, b'broken-revision-2-1', [b'parent-1', b'parent-2'])
    self.add_revision(repo, b'broken-revision-2-1', inv, [b'parent-2', b'parent-1'])
    self.versioned_root = repo.supports_rich_root()