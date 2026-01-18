import os
from breezy.tests import TestCaseWithTransport
def test_inventory_show_ids(self):
    expected = ''.join(('%-50s %s\n' % (path, file_id) for path, file_id in [('a', 'a-id'), ('b', 'b-id'), ('b/c', 'c-id')]))
    self.assertInventoryEqual(expected, '--show-ids')