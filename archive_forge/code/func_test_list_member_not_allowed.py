from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_list_member_not_allowed(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'pools/' + FAKE_PO + '/members', json=LIST_POLICY_ERR_RESP, status_code=403)
    ret = self.assertRaises(octavia.OctaviaClientException, self.api.member_list, FAKE_PO)
    self.assertEqual(POLICY_ERROR_STRING, ret.message)