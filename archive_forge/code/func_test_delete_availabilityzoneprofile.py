from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_availabilityzoneprofile(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'availabilityzoneprofiles/' + FAKE_AZPF, status_code=200)
    ret = self.api.availabilityzoneprofile_delete(FAKE_AZPF)
    self.assertEqual(200, ret.status_code)