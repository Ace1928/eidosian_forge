from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
def test_create_ephemeral_ispublic_defaults(self):
    f = self.cs.flavors.create('flavorcreate', 512, 1, 10, 1234)
    self.assert_request_id(f, fakes.FAKE_REQUEST_ID_LIST)
    body = self._create_body('flavorcreate', 512, 1, 10, 0, 1234, 0, 1.0, True)
    self.cs.assert_called('POST', '/flavors', body)
    self.assertIsInstance(f, self.flavor_type)