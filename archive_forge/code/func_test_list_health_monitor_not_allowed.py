from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_list_health_monitor_not_allowed(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'healthmonitors', json=LIST_POLICY_ERR_RESP, status_code=403)
    ret = self.assertRaises(octavia.OctaviaClientException, self.api.health_monitor_list)
    self.assertEqual(POLICY_ERROR_STRING, ret.message)