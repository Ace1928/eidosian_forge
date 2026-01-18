from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_inv_filter_dirs(self):
    inv = self.prepare_inv_with_nested_dirs()
    new_inv = inv.filter([b'doc-id', b'sub-id'])
    self.assertEqual([('', b'tree-root'), ('doc', b'doc-id'), ('src', b'src-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])