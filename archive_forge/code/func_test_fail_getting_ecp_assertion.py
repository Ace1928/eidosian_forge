import copy
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from keystoneauth1.tests.unit import utils
def test_fail_getting_ecp_assertion(self):
    self.requests_mock.get(self.TEST_URL, json={'version': fixture.V3Discovery(self.TEST_URL)}, headers={'Content-Type': 'application/json'})
    self.requests_mock.register_uri('POST', self.REQUEST_ECP_URL, status_code=401)
    self.assertRaises(exceptions.AuthorizationFailure, self.k2kplugin._get_ecp_assertion, self.session)