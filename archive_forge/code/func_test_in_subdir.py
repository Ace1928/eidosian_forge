import os
from breezy.tests import TestCaseWithTransport
def test_in_subdir(self):
    os.chdir('b')
    self.assertInventoryEqual('a\nb\nb/c\n')
    self.assertInventoryEqual('b\nb/c\n', '.')