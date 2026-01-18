from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_inv_filter_files(self):
    inv = self.prepare_inv_with_nested_dirs()
    new_inv = inv.filter([b'zz-id', b'hello-id', b'a-id'])
    self.assertEqual([('', b'tree-root'), ('src', b'src-id'), ('src/hello.c', b'hello-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id'), ('zz', b'zz-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])