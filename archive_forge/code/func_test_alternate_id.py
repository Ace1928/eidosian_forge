from openstack.network.v2 import quota
from openstack import resource
from openstack.tests.unit import base
def test_alternate_id(self):
    my_project_id = 'my-tenant-id'
    body = {'project_id': my_project_id, 'network': 12345}
    quota_obj = quota.Quota(**body)
    self.assertEqual(my_project_id, resource.Resource._get_id(quota_obj))