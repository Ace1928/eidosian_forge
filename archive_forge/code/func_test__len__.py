from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test__len__(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
    inv.get_entry(b'fileid').revision = b'filerev'
    inv.get_entry(b'fileid').executable = True
    inv.get_entry(b'fileid').text_sha1 = b'ffff'
    inv.get_entry(b'fileid').text_size = 1
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    self.assertEqual(2, len(chk_inv))