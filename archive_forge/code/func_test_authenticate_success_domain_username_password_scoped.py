from oslo_serialization import jsonutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def test_authenticate_success_domain_username_password_scoped(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    with self.deprecations.expect_deprecations_here():
        cs = client.Client(user_domain_name=self.TEST_DOMAIN_NAME, username=self.TEST_USER, password=self.TEST_TOKEN, project_id=self.TEST_TENANT_ID, auth_url=self.TEST_URL)
    self.assertEqual(cs.management_url, self.TEST_RESPONSE_DICT['token']['catalog'][3]['endpoints'][2]['url'])
    self.assertEqual(cs.auth_token, self.TEST_RESPONSE_HEADERS['X-Subject-Token'])