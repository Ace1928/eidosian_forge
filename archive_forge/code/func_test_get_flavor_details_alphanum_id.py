from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_get_flavor_details_alphanum_id(self):
    f = self.cs.flavors.get('aa1')
    self.assert_request_id(f, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/flavors/aa1')
    self.assertIsInstance(f, self.flavor_type)
    self.assertEqual(128, f.ram)
    self.assertEqual(0, f.disk)
    self.assertEqual(0, f.ephemeral)
    self.assertTrue(f.is_public)