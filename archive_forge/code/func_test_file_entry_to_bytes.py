from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_file_entry_to_bytes(self):
    inv = CHKInventory(None)
    ie = inventory.InventoryFile(b'file-id', 'filename', b'parent-id')
    ie.executable = True
    ie.revision = b'file-rev-id'
    ie.text_sha1 = b'abcdefgh'
    ie.text_size = 100
    bytes = inv._entry_to_bytes(ie)
    self.assertEqual(b'file: file-id\nparent-id\nfilename\nfile-rev-id\nabcdefgh\n100\nY', bytes)
    ie2 = inv._bytes_to_entry(bytes)
    self.assertEqual(ie, ie2)
    self.assertIsInstance(ie2.name, str)
    self.assertEqual((b'filename', b'file-id', b'file-rev-id'), inv._bytes_to_utf8name_key(bytes))