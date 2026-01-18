from openstack.block_storage.v2 import capabilities
from openstack.tests.unit import base
def test_make_capabilities(self):
    capabilities_resource = capabilities.Capabilities(**CAPABILITIES)
    self.assertEqual(CAPABILITIES['description'], capabilities_resource.description)
    self.assertEqual(CAPABILITIES['display_name'], capabilities_resource.display_name)
    self.assertEqual(CAPABILITIES['driver_version'], capabilities_resource.driver_version)
    self.assertEqual(CAPABILITIES['namespace'], capabilities_resource.namespace)
    self.assertEqual(CAPABILITIES['pool_name'], capabilities_resource.pool_name)
    self.assertEqual(CAPABILITIES['properties'], capabilities_resource.properties)
    self.assertEqual(CAPABILITIES['replication_targets'], capabilities_resource.replication_targets)
    self.assertEqual(CAPABILITIES['storage_protocol'], capabilities_resource.storage_protocol)
    self.assertEqual(CAPABILITIES['vendor_name'], capabilities_resource.vendor_name)
    self.assertEqual(CAPABILITIES['visibility'], capabilities_resource.visibility)
    self.assertEqual(CAPABILITIES['volume_backend_name'], capabilities_resource.volume_backend_name)