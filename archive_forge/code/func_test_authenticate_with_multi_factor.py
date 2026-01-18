import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_authenticate_with_multi_factor(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    passcode = '123456'
    auth = v3.MultiFactor(self.TEST_URL, auth_methods=['v3password', 'v3totp'], username=self.TEST_USER, password=self.TEST_PASS, passcode=passcode, user_domain_id=self.TEST_DOMAIN_ID, project_id=self.TEST_TENANT_ID)
    self.assertTrue(auth.has_scope_parameters)
    s = session.Session(auth=auth)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'identity': {'methods': ['password', 'totp'], 'totp': {'user': {'name': self.TEST_USER, 'passcode': passcode, 'domain': {'id': self.TEST_DOMAIN_ID}}}, 'password': {'user': {'name': self.TEST_USER, 'password': self.TEST_PASS, 'domain': {'id': self.TEST_DOMAIN_ID}}}}, 'scope': {'project': {'id': self.TEST_TENANT_ID}}}}
    self.assertRequestBodyIs(json=req)
    self.assertRequestHeaderEqual('Content-Type', 'application/json')
    self.assertRequestHeaderEqual('Accept', 'application/json')
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)