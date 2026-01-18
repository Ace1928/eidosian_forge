from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_directory_children_on_demand(self):
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
    lines = chk_inv.to_lines()
    new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
    root_entry = new_inv.get_entry(inv.root.file_id)
    self.assertEqual(None, root_entry._children)
    self.assertEqual({'file'}, set(root_entry.children))
    file_direct = new_inv.get_entry(b'fileid')
    file_found = root_entry.children['file']
    self.assertEqual(file_direct.kind, file_found.kind)
    self.assertEqual(file_direct.file_id, file_found.file_id)
    self.assertEqual(file_direct.parent_id, file_found.parent_id)
    self.assertEqual(file_direct.name, file_found.name)
    self.assertEqual(file_direct.revision, file_found.revision)
    self.assertEqual(file_direct.text_sha1, file_found.text_sha1)
    self.assertEqual(file_direct.text_size, file_found.text_size)
    self.assertEqual(file_direct.executable, file_found.executable)