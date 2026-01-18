from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def make_init_inventory(self):
    inv = inventory.Inventory(b'tree-root')
    inv.revision = b'initial-rev'
    inv.root.revision = b'initial-rev'
    return self.inv_to_test_inv(inv)