from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_load_balancer_error(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'loadbalancers/' + FAKE_LB, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.load_balancer_delete, FAKE_LB)