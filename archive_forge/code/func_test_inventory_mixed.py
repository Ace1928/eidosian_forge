import os
from breezy.tests import TestCaseWithTransport
def test_inventory_mixed(self):
    """Test that we get expected results when mixing parameters"""
    a_line = '%-50s %s\n' % ('a', 'a-id')
    b_line = '%-50s %s\n' % ('b', 'b-id')
    c_line = '%-50s %s\n' % ('b/c', 'c-id')
    self.assertInventoryEqual('', '--kind directory a')
    self.assertInventoryEqual(a_line + c_line, '--kind file --show-ids')
    self.assertInventoryEqual(c_line, '--kind file --show-ids b b/c')