import copy
import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from testtools import testcase
from keystoneclient import exceptions
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_authenticate_success_expired(self):
    resp_a = copy.deepcopy(self.TEST_RESPONSE_DICT)
    resp_b = copy.deepcopy(self.TEST_RESPONSE_DICT)
    headers = {'Content-Type': 'application/json'}
    resp_a['access']['token']['expires'] = (timeutils.utcnow() - datetime.timedelta(1)).isoformat()
    TEST_TOKEN = 'abcdef'
    resp_b['access']['token']['expires'] = '2999-01-01T00:00:10.000123Z'
    resp_b['access']['token']['id'] = TEST_TOKEN
    self.stub_auth(response_list=[{'json': resp_a, 'headers': headers}, {'json': resp_b, 'headers': headers}])
    with self.deprecations.expect_deprecations_here():
        cs = client.Client(project_id=self.TEST_TENANT_ID, auth_url=self.TEST_URL, username=self.TEST_USER, password=self.TEST_TOKEN)
    self.assertEqual(cs.management_url, self.TEST_RESPONSE_DICT['access']['serviceCatalog'][3]['endpoints'][0]['adminURL'])
    with self.deprecations.expect_deprecations_here():
        self.assertEqual(cs.auth_token, TEST_TOKEN)
    self.assertRequestBodyIs(json=self.TEST_REQUEST_BODY)