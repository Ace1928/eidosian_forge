import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_inventory_list(self):
    rp_uuid = self.rp['uuid']
    updates = {'min_unit': 1, 'max_unit': 12, 'reserved': 0, 'step_size': 1, 'total': 12, 'allocation_ratio': 16.0}
    expected = [updates.copy()]
    expected[0]['resource_class'] = 'VCPU'
    expected[0]['used'] = 0
    args = ['VCPU:%s=%s' % (k, v) for k, v in updates.items()]
    self.resource_inventory_set(rp_uuid, *args)
    self.assertEqual(expected, self.resource_inventory_list(rp_uuid, include_used=True))