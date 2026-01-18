from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_find_id(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks/1', json={'network': self.FAKE_NETWORK_RESP}, status_code=200)
    ret = self.api.network_find('1')
    self.assertEqual(self.FAKE_NETWORK_RESP, ret)