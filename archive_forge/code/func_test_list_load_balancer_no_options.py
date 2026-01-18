from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_list_load_balancer_no_options(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'loadbalancers', json=LIST_LB_RESP, status_code=200)
    ret = self.api.load_balancer_list()
    self.assertEqual(LIST_LB_RESP, ret)