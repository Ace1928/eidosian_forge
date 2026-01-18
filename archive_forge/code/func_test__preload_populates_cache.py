from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test__preload_populates_cache(self):
    inv = Inventory()
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    root_id = inv.root.file_id
    inv.add(InventoryFile(b'fileid', 'file', root_id))
    inv.get_entry(b'fileid').revision = b'filerev'
    inv.get_entry(b'fileid').executable = True
    inv.get_entry(b'fileid').text_sha1 = b'ffff'
    inv.get_entry(b'fileid').text_size = 1
    inv.add(InventoryDirectory(b'dirid', 'dir', root_id))
    inv.get_entry(b'dirid').revision = b'dirrev'
    inv.add(InventoryFile(b'childid', 'child', b'dirid'))
    inv.get_entry(b'childid').revision = b'filerev'
    inv.get_entry(b'childid').executable = False
    inv.get_entry(b'childid').text_sha1 = b'dddd'
    inv.get_entry(b'childid').text_size = 1
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
    lines = chk_inv.to_lines()
    new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
    self.assertEqual({}, new_inv._fileid_to_entry_cache)
    self.assertFalse(new_inv._fully_cached)
    new_inv._preload_cache()
    self.assertEqual(sorted([root_id, b'fileid', b'dirid', b'childid']), sorted(new_inv._fileid_to_entry_cache.keys()))
    self.assertTrue(new_inv._fully_cached)
    ie_root = new_inv._fileid_to_entry_cache[root_id]
    self.assertEqual(['dir', 'file'], sorted(ie_root._children.keys()))
    ie_dir = new_inv._fileid_to_entry_cache[b'dirid']
    self.assertEqual(['child'], sorted(ie_dir._children.keys()))