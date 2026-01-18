from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_delete_name(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks/label1', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks', json={'networks': self.LIST_NETWORK_RESP}, status_code=200)
    self.requests_mock.register_uri('DELETE', FAKE_URL + '/os-networks/1', status_code=202)
    ret = self.api.network_delete('label1')
    self.assertEqual(202, ret.status_code)
    self.assertEqual('', ret.text)