import os
from breezy.tests import TestCaseWithTransport
def test_inventory_specific_files(self):
    self.assertInventoryEqual('a\n', 'a')
    self.assertInventoryEqual('b\nb/c\n', 'b b/c')
    self.assertInventoryEqual('b\nb/c\n', 'b')