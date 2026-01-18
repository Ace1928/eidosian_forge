import os
from breezy.tests import TestCaseWithTransport
def test_inventory(self):
    self.assertInventoryEqual('a\nb\nb/c\n')