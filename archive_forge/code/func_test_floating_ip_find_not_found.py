from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_floating_ip_find_not_found(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-floating-ips/1.2.3.4', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-floating-ips', json={'floating_ips': self.LIST_FLOATING_IP_RESP}, status_code=200)
    self.assertRaises(osc_lib_exceptions.NotFound, self.api.floating_ip_find, '1.2.3.4')