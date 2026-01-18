from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_list_body(self):
    self.requests_mock.register_uri('POST', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
    ret = self.api.list('qaz', body=api_fakes.LIST_BODY)
    self.assertEqual(api_fakes.LIST_RESP, ret)