from oslo_serialization import jsonutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def test_authenticate_success(self):
    TEST_TOKEN = 'abcdef'
    ident = self.TEST_REQUEST_BODY['auth']['identity']
    del ident['password']['user']['domain']
    del ident['password']['user']['name']
    ident['password']['user']['id'] = self.TEST_USER
    self.stub_auth(json=self.TEST_RESPONSE_DICT, subject_token=TEST_TOKEN)
    with self.deprecations.expect_deprecations_here():
        cs = client.Client(user_id=self.TEST_USER, password=self.TEST_TOKEN, project_id=self.TEST_TENANT_ID, auth_url=self.TEST_URL)
    self.assertEqual(cs.auth_token, TEST_TOKEN)
    self.assertRequestBodyIs(json=self.TEST_REQUEST_BODY)