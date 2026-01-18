from openstack.shared_file_system.v2 import storage_pool
from openstack.tests.unit import base
def test_make_storage_pool(self):
    pool_resource = storage_pool.StoragePool(**EXAMPLE)
    self.assertEqual(EXAMPLE['pool'], pool_resource.pool)
    self.assertEqual(EXAMPLE['host'], pool_resource.host)
    self.assertEqual(EXAMPLE['name'], pool_resource.name)
    self.assertEqual(EXAMPLE['backend'], pool_resource.backend)
    self.assertEqual(EXAMPLE['capabilities'], pool_resource.capabilities)