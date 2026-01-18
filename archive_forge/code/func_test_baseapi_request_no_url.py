from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_request_no_url(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=200)
    self.assertRaises(ksa_exceptions.EndpointNotFound, self.api._request, 'GET', '')
    self.assertIsNotNone(self.api.session)
    self.assertNotEqual(self.sess, self.api.session)