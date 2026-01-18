from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_l7rule_error(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules', text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.l7rule_create, FAKE_L7PO, json=SINGLE_L7RU_RESP)