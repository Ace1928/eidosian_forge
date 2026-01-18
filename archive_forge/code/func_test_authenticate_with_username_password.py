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
def test_authenticate_with_username_password(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)
    self.assertFalse(a.has_scope_parameters)
    s = session.Session(auth=a)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    req = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.TEST_USER, 'password': self.TEST_PASS}}}}}
    self.assertRequestBodyIs(json=req)
    self.assertRequestHeaderEqual('Content-Type', 'application/json')
    self.assertRequestHeaderEqual('Accept', 'application/json')
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)