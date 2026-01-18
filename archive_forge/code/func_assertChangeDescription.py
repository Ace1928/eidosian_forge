from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def assertChangeDescription(self, expected_change, old_ie, new_ie):
    change = InventoryEntry.describe_change(old_ie, new_ie)
    self.assertEqual(expected_change, change)