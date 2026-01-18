from openstack.block_storage.v3 import resource_filter
from openstack.tests.unit import base
def test_make_resource_filter(self):
    resource = resource_filter.ResourceFilter(**RESOURCE_FILTER)
    self.assertEqual(RESOURCE_FILTER['filters'], resource.filters)
    self.assertEqual(RESOURCE_FILTER['resource'], resource.resource)