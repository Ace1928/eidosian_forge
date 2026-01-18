import uuid
from osc_placement.tests.functional import base
def test_list_multiple(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create()
    self.resource_inventory_set(rp1['uuid'], 'MEMORY_MB=8192', 'DISK_GB=512')
    self.resource_inventory_set(rp2['uuid'], 'MEMORY_MB=16384', 'DISK_GB=1024')
    candidates = self.allocation_candidate_list(resources=('MEMORY_MB=1024', 'DISK_GB=80'))
    rps = {c['resource provider']: c for c in candidates}
    self.assertResourceEqual('MEMORY_MB=1024,DISK_GB=80', rps[rp1['uuid']]['allocation'])
    self.assertResourceEqual('MEMORY_MB=1024,DISK_GB=80', rps[rp2['uuid']]['allocation'])
    self.assertResourceEqual('MEMORY_MB=0/8192,DISK_GB=0/512', rps[rp1['uuid']]['inventory used/capacity'])
    self.assertResourceEqual('MEMORY_MB=0/16384,DISK_GB=0/1024', rps[rp2['uuid']]['inventory used/capacity'])