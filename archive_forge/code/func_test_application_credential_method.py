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
def test_application_credential_method(self):
    self.stub_auth(json=self.TEST_APP_CRED_TOKEN_RESPONSE)
    ac = v3.ApplicationCredential(self.TEST_URL, application_credential_id=self.TEST_APP_CRED_ID, application_credential_secret=self.TEST_APP_CRED_SECRET)
    req = {'auth': {'identity': {'methods': ['application_credential'], 'application_credential': {'id': self.TEST_APP_CRED_ID, 'secret': self.TEST_APP_CRED_SECRET}}}}
    s = session.Session(auth=ac)
    self.assertEqual({'X-Auth-Token': self.TEST_TOKEN}, s.get_auth_headers())
    self.assertRequestBodyIs(json=req)
    self.assertEqual(s.auth.auth_ref.auth_token, self.TEST_TOKEN)