from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_availabilityzoneprofile(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'availabilityzoneprofiles', json=SINGLE_AZPF_RESP, status_code=200)
    ret = self.api.availabilityzoneprofile_create(json=SINGLE_AZPF_RESP)
    self.assertEqual(SINGLE_AZPF_RESP, ret)