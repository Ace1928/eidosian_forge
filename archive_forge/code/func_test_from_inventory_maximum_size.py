from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_from_inventory_maximum_size(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv, 120)
    chk_inv.id_to_entry._ensure_root()
    self.assertEqual(120, chk_inv.id_to_entry._root_node.maximum_size)
    self.assertEqual(1, chk_inv.id_to_entry._root_node._key_width)
    p_id_basename = chk_inv.parent_id_basename_to_file_id
    p_id_basename._ensure_root()
    self.assertEqual(120, p_id_basename._root_node.maximum_size)
    self.assertEqual(2, p_id_basename._root_node._key_width)