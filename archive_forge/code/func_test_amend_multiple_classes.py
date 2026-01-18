import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_amend_multiple_classes(self):
    rp = self.resource_provider_create()
    resp = self.resource_inventory_set(rp['uuid'], 'VCPU=8', 'VCPU:max_unit=4', 'MEMORY_MB=1024', 'MEMORY_MB:reserved=256', 'DISK_GB=16', 'DISK_GB:allocation_ratio=1.5', 'DISK_GB:min_unit=2', 'DISK_GB:step_size=2', amend=True)

    def check(inventories):
        self.assertEqual(8, inventories['VCPU']['total'])
        self.assertEqual(4, inventories['VCPU']['max_unit'])
        self.assertEqual(1024, inventories['MEMORY_MB']['total'])
        self.assertEqual(256, inventories['MEMORY_MB']['reserved'])
        self.assertEqual(16, inventories['DISK_GB']['total'])
        self.assertEqual(1.5, inventories['DISK_GB']['allocation_ratio'])
        self.assertEqual(2, inventories['DISK_GB']['min_unit'])
        self.assertEqual(2, inventories['DISK_GB']['step_size'])
    inventories = {r['resource_class']: r for r in resp}
    check(inventories)
    resp = self.resource_inventory_list(rp['uuid'])
    inventories = {r['resource_class']: r for r in resp}
    check(inventories)
    resp = self.resource_inventory_set(rp['uuid'], 'VCPU:allocation_ratio=5.0', amend=True)
    inventories = {r['resource_class']: r for r in resp}
    check(inventories)
    self.assertEqual(5.0, inventories['VCPU']['allocation_ratio'])
    resp = self.resource_inventory_list(rp['uuid'])
    inventories = {r['resource_class']: r for r in resp}
    check(inventories)
    self.assertEqual(5.0, inventories['VCPU']['allocation_ratio'])