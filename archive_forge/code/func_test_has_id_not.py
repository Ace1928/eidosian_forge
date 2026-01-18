from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_has_id_not(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    self.assertFalse(chk_inv.has_id(b'fileid'))