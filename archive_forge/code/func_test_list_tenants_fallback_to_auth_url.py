import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_list_tenants_fallback_to_auth_url(self):
    new_auth_url = 'http://keystone.test:5000/v2.0'
    token = fixture.V2Token(token_id=self.TEST_TOKEN, user_name=self.TEST_USER, user_id=self.TEST_USER_ID)
    self.stub_auth(base_url=new_auth_url, json=token)
    self.stub_url('GET', ['tenants'], base_url=new_auth_url, json=self.TEST_TENANTS)
    with self.deprecations.expect_deprecations_here():
        c = client.Client(username=self.TEST_USER, auth_url=new_auth_url, password=uuid.uuid4().hex)
    self.assertIsNone(c.management_url)
    tenant_list = c.tenants.list()
    [self.assertIsInstance(t, tenants.Tenant) for t in tenant_list]
    self.assertEqual(len(self.TEST_TENANTS['tenants']['values']), len(tenant_list))