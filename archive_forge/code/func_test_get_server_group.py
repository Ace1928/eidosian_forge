from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_get_server_group(self):
    id = '2cbd51f4-fafe-4cdb-801b-cf913a6f288b'
    server_group = self.cs.server_groups.get(id)
    self.assert_request_id(server_group, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-server-groups/%s' % id)
    self.assertIsInstance(server_group, server_groups.ServerGroup)