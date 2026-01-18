from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavor_access
def test_list_access_by_flavor_private(self):
    kwargs = {'flavor': self.cs.flavors.get(2)}
    r = self.cs.flavor_access.list(**kwargs)
    self.assert_request_id(r, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/flavors/2/os-flavor-access')
    for a in r:
        self.assertIsInstance(a, flavor_access.FlavorAccess)