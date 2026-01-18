from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_get_flavor_details_diablo(self):
    f = self.cs.flavors.get(3)
    self.assert_request_id(f, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/flavors/3')
    self.assertIsInstance(f, self.flavor_type)
    self.assertEqual(256, f.ram)
    self.assertEqual(10, f.disk)
    self.assertEqual('N/A', f.ephemeral)
    self.assertEqual('N/A', f.is_public)