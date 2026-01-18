import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_list_tenants_use_admin_url(self):
    self.stub_url('GET', ['tenants'], json=self.TEST_TENANTS)
    tenant_list = self.client.tenants.list()
    self.assertEqual(self.TEST_URL + '/tenants', self.requests_mock.last_request.url)
    [self.assertIsInstance(t, tenants.Tenant) for t in tenant_list]
    self.assertEqual(len(self.TEST_TENANTS['tenants']['values']), len(tenant_list))