from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_load_balancer_503_error(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'loadbalancers', status_code=503)
    self.assertRaisesRegex(exceptions.OctaviaClientException, 'Service Unavailable', self.api.load_balancer_create, json=SINGLE_LB_RESP)