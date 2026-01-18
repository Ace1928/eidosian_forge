import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_inventory_delete_not_found(self):
    exc = self.assertRaises(base.CommandException, self.resource_inventory_delete, self.rp['uuid'], 'VCPU')
    self.assertIn('No inventory of class VCPU found for delete', str(exc))