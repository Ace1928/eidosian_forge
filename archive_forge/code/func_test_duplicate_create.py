import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def test_duplicate_create(self):
    req_body = {'tenant': {'name': 'tenantX', 'description': 'The duplicate tenant.', 'enabled': True}}
    resp_body = {'error': {'message': 'Conflict occurred attempting to store project.', 'code': 409, 'title': 'Conflict'}}
    self.stub_url('POST', ['tenants'], status_code=409, json=resp_body)

    def create_duplicate_tenant():
        self.client.tenants.create(req_body['tenant']['name'], req_body['tenant']['description'], req_body['tenant']['enabled'])
    self.assertRaises(exceptions.Conflict, create_duplicate_tenant)