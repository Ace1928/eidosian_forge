from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def file_parents(self, repo, revision_id):
    key = (b'a-file-id', revision_id)
    parent_map = repo.texts.get_parent_map([key])
    return tuple((parent[-1] for parent in parent_map[key]))