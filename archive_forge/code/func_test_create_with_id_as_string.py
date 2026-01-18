from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_create_with_id_as_string(self):
    flavor_id = 'foobar'
    f = self.cs.flavors.create('flavorcreate', 512, 1, 10, flavor_id, ephemeral=10, is_public=False)
    self.assert_request_id(f, fakes.FAKE_REQUEST_ID_LIST)
    body = self._create_body('flavorcreate', 512, 1, 10, 10, flavor_id, 0, 1.0, False)
    self.cs.assert_called('POST', '/flavors', body)
    self.assertIsInstance(f, self.flavor_type)