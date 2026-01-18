import uuid
from osc_placement.tests.functional import base
def test_list_one(self):
    rp = self.resource_provider_create()
    self.resource_inventory_set(rp['uuid'], 'MEMORY_MB=1024')
    candidates = self.allocation_candidate_list(resources=('MEMORY_MB=256',))
    self.assertIn(rp['uuid'], [candidate['resource provider'] for candidate in candidates])