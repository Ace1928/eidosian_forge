from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def make_link(self, file_id, name, parent_id, target='link-target\n'):
    ie = InventoryLink(file_id, name, parent_id)
    ie.symlink_target = target
    return ie