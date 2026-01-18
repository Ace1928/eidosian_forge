import uuid
from osc_placement.tests.functional import base
def test_allocation_unset_provider_and_rc(self):
    """Tests removing allocations of resource classes for a provider ."""
    updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, provider=self.rp3['uuid'], resource_class=['VCPU', 'MEMORY_MB'])
    expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VGPU': 1}}, {'resource_provider': self.rp4['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1, 'MEMORY_MB': 256}}]
    self.assertEqual(expected, updated_allocs)