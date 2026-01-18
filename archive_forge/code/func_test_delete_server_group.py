import uuid
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_group(self):
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-server-groups']), json={'server_groups': [self.fake_group]}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['os-server-groups', self.group_id]), json={'server_groups': [self.fake_group]})])
    self.assertTrue(self.cloud.delete_server_group(self.group_name))
    self.assert_calls()