from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_availabilityzone(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'availabilityzones', json=SINGLE_AZ_RESP, status_code=200)
    ret = self.api.availabilityzone_create(json=SINGLE_AZ_RESP)
    self.assertEqual(SINGLE_AZ_RESP, ret)