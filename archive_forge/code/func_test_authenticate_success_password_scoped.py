import copy
import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_authenticate_success_password_scoped(self):
    self.stub_auth(json=self.TEST_RESPONSE_DICT)
    with self.deprecations.expect_deprecations_here():
        cs = client.Client(username=self.TEST_USER, password=self.TEST_TOKEN, project_id=self.TEST_TENANT_ID, auth_url=self.TEST_URL)
    self.assertEqual(cs.management_url, self.TEST_RESPONSE_DICT['access']['serviceCatalog'][3]['endpoints'][0]['adminURL'])
    self.assertEqual(cs.auth_token, self.TEST_RESPONSE_DICT['access']['token']['id'])
    self.assertRequestBodyIs(json=self.TEST_REQUEST_BODY)