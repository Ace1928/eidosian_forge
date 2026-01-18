import uuid
from osc_placement.tests.functional import base
def test_allocation_unset_one_provider(self):
    """Tests removing allocations for one specific provider."""
    updated_allocs = self.resource_allocation_unset(self.consumer_uuid1, provider=self.rp1['uuid'])
    expected = [{'resource_provider': self.rp2['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}]
    self.assertEqual(expected, updated_allocs)