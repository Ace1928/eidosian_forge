from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_is_root(self):
    """Ensure our root-checking code is accurate."""
    inv = self.make_init_inventory()
    self.assertTrue(inv.is_root(b'tree-root'))
    self.assertFalse(inv.is_root(b'booga'))
    ie = inv.get_entry(b'tree-root').copy()
    ie.file_id = b'booga'
    inv = inv.create_by_apply_delta([('', None, b'tree-root', None), (None, '', b'booga', ie)], b'new-rev-2')
    self.assertFalse(inv.is_root(b'TREE_ROOT'))
    self.assertTrue(inv.is_root(b'booga'))