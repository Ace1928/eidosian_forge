from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_floating_ip_create_not_found(self):
    self.requests_mock.register_uri('POST', FAKE_URL + '/os-floating-ips', status_code=404)
    self.assertRaises(osc_lib_exceptions.NotFound, self.api.floating_ip_create, 'not-nova')