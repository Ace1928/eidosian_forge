import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_delete_via_set(self):
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'DISK_GB=16')
    self.resource_inventory_set(rp['uuid'])
    self.assertEqual([], self.resource_inventory_list(rp['uuid']))