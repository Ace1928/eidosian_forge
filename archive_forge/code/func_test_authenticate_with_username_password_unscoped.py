import argparse
import copy
from unittest import mock
import uuid
from keystoneauth1 import fixture
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient.auth.identity.v3 import base as v3_base
from keystoneclient import client
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_authenticate_with_username_password_unscoped(self):
    del self.TEST_RESPONSE_DICT['token']['catalog']
    del self.TEST_RESPONSE_DICT['token']['project']
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    self.stub_url(method='GET', json=self.TEST_DISCOVERY_RESPONSE)
    test_user_id = self.TEST_RESPONSE_DICT['token']['user']['id']
    self.stub_url(method='GET', json=self.TEST_PROJECTS_RESPONSE, parts=['users', test_user_id, 'projects'])
    a = v3.Password(self.TEST_URL, username=self.TEST_USER, password=self.TEST_PASS)
    s = session.Session(auth=a)
    cs = client.Client(session=s)
    self.assertEqual(test_user_id, a.auth_ref.user_id)
    t = cs.projects.list(user=a.auth_ref.user_id)
    self.assertEqual(2, len(t))