from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_deserialise_gives_CHKInventory(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
    self.assertEqual(b'revid', new_inv.revision_id)
    self.assertEqual('directory', new_inv.root.kind)
    self.assertEqual(inv.root.file_id, new_inv.root.file_id)
    self.assertEqual(inv.root.parent_id, new_inv.root.parent_id)
    self.assertEqual(inv.root.name, new_inv.root.name)
    self.assertEqual(b'rootrev', new_inv.root.revision)
    self.assertEqual(b'plain', new_inv._search_key_name)