import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
def test_base(self):
    self.assertAdduserBranchContent('base')