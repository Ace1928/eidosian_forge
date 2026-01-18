from openstack.shared_file_system.v2 import availability_zone as az
from openstack.tests.unit import base
def test_make_availability_zone(self):
    az_resource = az.AvailabilityZone(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], az_resource.id)
    self.assertEqual(EXAMPLE['name'], az_resource.name)
    self.assertEqual(EXAMPLE['created_at'], az_resource.created_at)
    self.assertEqual(EXAMPLE['updated_at'], az_resource.updated_at)