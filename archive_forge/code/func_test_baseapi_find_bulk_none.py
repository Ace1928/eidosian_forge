from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_find_bulk_none(self):
    self.requests_mock.register_uri('GET', self.BASE_URL + '/qaz', json=api_fakes.LIST_RESP, status_code=200)
    ret = self.api.find_bulk('qaz')
    self.assertEqual(api_fakes.LIST_RESP, ret)
    ret = self.api.find_bulk('qaz', headers={})
    self.assertEqual(api_fakes.LIST_RESP, ret)