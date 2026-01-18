from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_error_encoding(self):
    inv = inventory.Inventory(b'tree-root')
    inv.add(InventoryFile(b'a-id', 'ሴ', b'tree-root'))
    e = self.assertRaises(errors.InconsistentDelta, inv.add, InventoryFile(b'b-id', 'ሴ', b'tree-root'))
    self.assertContainsRe(str(e), '\\u1234')