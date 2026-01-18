import copy
import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_auth_url_token_authentication(self):
    fake_token = 'fake_token'
    fake_url = '/fake-url'
    fake_resp = {'result': True}
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    self.stub_url('GET', [fake_url], json=fake_resp, base_url=self.TEST_ADMIN_IDENTITY_ENDPOINT)
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(auth_url=self.TEST_URL, token=fake_token)
    json_body = jsonutils.loads(self.requests_mock.last_request.body)
    self.assertEqual(json_body['auth']['token']['id'], fake_token)
    with self.deprecations.expect_deprecations_here():
        resp, body = cl.get(fake_url)
    self.assertEqual(fake_resp, body)
    token = self.requests_mock.last_request.headers.get('X-Auth-Token')
    self.assertEqual(self.TEST_TOKEN, token)