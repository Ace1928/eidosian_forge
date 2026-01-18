import os
from breezy.tests import TestCaseWithTransport
def test_inventory_kind(self):
    self.assertInventoryEqual('a\nb/c\n', '--kind file')
    self.assertInventoryEqual('b\n', '--kind directory')