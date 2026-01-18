from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_availabilityzone(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'availabilityzones/' + FAKE_AZ, status_code=200)
    ret = self.api.availabilityzone_delete(FAKE_AZ)
    self.assertEqual(200, ret.status_code)