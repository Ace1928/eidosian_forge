from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_add_new_id_existing_path(self):
    inv = self.get_empty_inventory()
    parent1 = inventory.InventoryDirectory(b'p-1', 'dir1', inv.root.file_id)
    parent1.revision = b'result'
    parent2 = inventory.InventoryDirectory(b'p-2', 'dir1', inv.root.file_id)
    parent2.revision = b'result'
    inv.add(parent1)
    delta = [(None, 'dir1', b'p-2', parent2)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)