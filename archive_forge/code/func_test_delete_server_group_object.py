from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_delete_server_group_object(self):
    id = '2cbd51f4-fafe-4cdb-801b-cf913a6f288b'
    server_group = self.cs.server_groups.get(id)
    ret = server_group.delete()
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('DELETE', '/os-server-groups/%s' % id)