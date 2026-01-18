import uuid
from osc_placement.tests.functional import base
def test_allocation_unset_one_resource_class(self):
    """Tests removing allocations for resource classes."""
    updated_allocs = self.resource_allocation_unset(self.consumer_uuid2, resource_class=['MEMORY_MB'])
    expected = [{'resource_provider': self.rp3['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1, 'VGPU': 1}}, {'resource_provider': self.rp4['uuid'], 'generation': 3, 'project_id': self.project_uuid, 'user_id': self.user_uuid, 'resources': {'VCPU': 1}}]
    self.assertEqual(expected, updated_allocs)