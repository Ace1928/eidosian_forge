import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_inventory_delete(self):
    rp_uuid = self.rp['uuid']
    self.resource_inventory_set(rp_uuid, 'VCPU=8')
    self.resource_inventory_delete(rp_uuid, 'VCPU')
    exc = self.assertRaises(base.CommandException, self.resource_inventory_show, rp_uuid, 'VCPU')
    self.assertIn('No inventory of class VCPU for {}'.format(rp_uuid), str(exc))