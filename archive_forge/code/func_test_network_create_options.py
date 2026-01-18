from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_create_options(self):
    self.requests_mock.register_uri('POST', FAKE_URL + '/os-networks', json={'network': self.FAKE_NETWORK_RESP}, status_code=200)
    ret = self.api.network_create(name='label1', subnet='1.2.3.0/24')
    self.assertEqual(self.FAKE_NETWORK_RESP, ret)