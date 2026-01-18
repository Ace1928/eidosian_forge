from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_groups as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_groups
def test_find_no_existing_server_groups_by_name(self):
    expected_name = 'no-exist'
    kwargs = {self.cs.server_groups.resource_class.NAME_ATTR: expected_name}
    self.assertRaises(exceptions.NotFound, self.cs.server_groups.find, **kwargs)
    self.assert_called('GET', '/os-server-groups')