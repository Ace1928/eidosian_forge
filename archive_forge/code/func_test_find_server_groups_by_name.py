from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_find_server_groups_by_name(self):
    expected_name = 'ig1'
    kwargs = {self.cs.server_groups.resource_class.NAME_ATTR: expected_name}
    server_group = self.cs.server_groups.find(**kwargs)
    self.assert_request_id(server_group, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-server-groups')
    self.assertIsInstance(server_group, server_groups.ServerGroup)
    actual_name = getattr(server_group, self.cs.server_groups.resource_class.NAME_ATTR)
    self.assertEqual(expected_name, actual_name)