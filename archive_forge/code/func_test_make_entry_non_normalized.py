from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_make_entry_non_normalized(self):
    orig_normalized_filename = osutils.normalized_filename
    try:
        osutils.normalized_filename = osutils._accessible_normalized_filename
        entry = inventory.make_entry('file', 'å', ROOT_ID)
        self.assertEqual('å', entry.name)
        self.assertIsInstance(entry, inventory.InventoryFile)
        osutils.normalized_filename = osutils._inaccessible_normalized_filename
        self.assertRaises(errors.InvalidNormalization, inventory.make_entry, 'file', 'å', ROOT_ID)
    finally:
        osutils.normalized_filename = orig_normalized_filename