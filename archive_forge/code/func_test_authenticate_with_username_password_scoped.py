import copy
import json
import time
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v2
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_authenticate_with_username_password_scoped(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v2.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS, tenant_id=self.TEST_TENANT_ID)
    self.assertTrue(a.has_scope_parameters)
    self.assertIsNone(a.user_id)
    s = session.Session(a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'passwordCredentials': {'username': self.TEST_USER, 'password': self.TEST_PASS}, 'tenantId': self.TEST_TENANT_ID}}
    self.assertRequestBodyIs(json=req)
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)