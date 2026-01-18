from ..workingtree import InventoryModified
from . import TestCase
def test_inventory_modified(self):
    error = InventoryModified('a tree to be repred')
    self.assertEqualDiff("The current inventory for the tree 'a tree to be repred' has been modified, so a clean inventory cannot be read without data loss.", str(error))