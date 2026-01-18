from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_health_monitor(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'healthmonitors/' + FAKE_HM, status_code=200)
    ret = self.api.health_monitor_delete(FAKE_HM)
    self.assertEqual(200, ret.status_code)