from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_availabilityzoneprofile_error(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'availabilityzoneprofiles/' + FAKE_AZPF, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.availabilityzoneprofile_delete, FAKE_AZPF)