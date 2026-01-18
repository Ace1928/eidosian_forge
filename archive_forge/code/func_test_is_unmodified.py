from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_is_unmodified(self):
    f1 = self.make_file(b'file-id', 'file', b'tree-root')
    f1.revision = b'rev'
    self.assertTrue(f1.is_unmodified(f1))
    f2 = self.make_file(b'file-id', 'file', b'tree-root')
    f2.revision = b'rev'
    self.assertTrue(f1.is_unmodified(f2))
    f3 = self.make_file(b'file-id', 'file', b'tree-root')
    self.assertFalse(f1.is_unmodified(f3))
    f4 = self.make_file(b'file-id', 'file', b'tree-root')
    f4.revision = b'rev1'
    self.assertFalse(f1.is_unmodified(f4))