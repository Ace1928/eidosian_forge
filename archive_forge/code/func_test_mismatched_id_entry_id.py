from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_mismatched_id_entry_id(self):
    inv = self.get_empty_inventory()
    file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
    file1.revision = b'result'
    file1.text_size = 0
    file1.text_sha1 = b''
    delta = [(None, 'path', b'id', file1)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)