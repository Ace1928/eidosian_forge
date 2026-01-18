from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_floating_ip_add_id(self):
    self.requests_mock.register_uri('POST', FAKE_URL + '/servers/1/action', json={'server': {}}, status_code=200)
    self.requests_mock.register_uri('GET', FAKE_URL + '/servers/1', json={'server': self.FAKE_SERVER_RESP_1}, status_code=200)
    ret = self.api.floating_ip_add('1', '1.0.1.0')
    self.assertEqual(200, ret.status_code)