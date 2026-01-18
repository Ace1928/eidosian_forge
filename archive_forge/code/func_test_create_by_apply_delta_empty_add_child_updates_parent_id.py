from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_create_by_apply_delta_empty_add_child_updates_parent_id(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    chk_bytes = self.get_chk_bytes()
    base_inv = CHKInventory.from_inventory(chk_bytes, inv)
    a_entry = InventoryFile(b'A-id', 'A', inv.root.file_id)
    a_entry.revision = b'filerev'
    a_entry.executable = True
    a_entry.text_sha1 = b'ffff'
    a_entry.text_size = 1
    inv.add(a_entry)
    inv.revision_id = b'expectedid'
    reference_inv = CHKInventory.from_inventory(chk_bytes, inv)
    delta = [(None, 'A', b'A-id', a_entry)]
    new_inv = base_inv.create_by_apply_delta(delta, b'expectedid')
    reference_inv.id_to_entry._ensure_root()
    reference_inv.parent_id_basename_to_file_id._ensure_root()
    new_inv.id_to_entry._ensure_root()
    new_inv.parent_id_basename_to_file_id._ensure_root()
    self.assertEqual(reference_inv.revision_id, new_inv.revision_id)
    self.assertEqual(reference_inv.root_id, new_inv.root_id)
    self.assertEqual(reference_inv.id_to_entry._root_node._key, new_inv.id_to_entry._root_node._key)
    self.assertEqual(reference_inv.parent_id_basename_to_file_id._root_node._key, new_inv.parent_id_basename_to_file_id._root_node._key)