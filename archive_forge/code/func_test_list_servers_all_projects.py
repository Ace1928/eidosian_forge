import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_servers_all_projects(self):
    """This test verifies that when list_servers is called with
        `all_projects=True` that it passes `all_tenants=True` to nova."""
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['all_tenants=True']), complete_qs=True, json={'servers': []})])
    self.cloud.list_servers(all_projects=True)
    self.assert_calls()