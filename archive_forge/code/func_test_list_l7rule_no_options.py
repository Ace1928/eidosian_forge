from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_list_l7rule_no_options(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules', json=LIST_L7RU_RESP, status_code=200)
    ret = self.api.l7rule_list(FAKE_L7PO)
    self.assertEqual(LIST_L7RU_RESP, ret)