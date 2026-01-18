import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_inventory_show_not_found(self):
    rp_uuid = self.rp['uuid']
    exc = self.assertRaises(base.CommandException, self.resource_inventory_show, rp_uuid, 'VCPU')
    self.assertIn('No inventory of class VCPU for {}'.format(rp_uuid), str(exc))