from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test__getitems(self):
    inv = self.make_simple_inventory()
    self.assert_Getitems([b'dir1-id'], inv, [b'dir1-id'])
    self.assertTrue(b'dir1-id' in inv._fileid_to_entry_cache)
    self.assertFalse(b'sub-file2-id' in inv._fileid_to_entry_cache)
    self.assert_Getitems([b'dir1-id'], inv, [b'dir1-id'])
    self.assert_Getitems([b'dir1-id', b'sub-file2-id'], inv, [b'dir1-id', b'sub-file2-id'])
    self.assertTrue(b'dir1-id' in inv._fileid_to_entry_cache)
    self.assertTrue(b'sub-file2-id' in inv._fileid_to_entry_cache)