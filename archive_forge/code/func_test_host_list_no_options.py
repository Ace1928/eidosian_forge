from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_host_list_no_options(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-hosts', json={'hosts': self.LIST_HOST_RESP}, status_code=200)
    ret = self.api.host_list()
    self.assertEqual(self.LIST_HOST_RESP, ret)