from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_request_url_path(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.RESP_ITEM_1, status_code=200)
    ret = self.api._request('GET', '/qaz')
    self.assertEqual(api_fakes.RESP_ITEM_1, ret.json())
    self.assertIsNotNone(self.api.session)
    self.assertEqual(self.sess, self.api.session)