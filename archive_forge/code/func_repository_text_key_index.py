from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def repository_text_key_index(self):
    result = {}
    if self.versioned_root:
        result.update(self.versioned_repository_text_keys())
    result.update(self.repository_text_keys())
    return result