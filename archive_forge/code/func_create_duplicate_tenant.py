import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import tenants
from keystoneclient.v2_0 import users
def create_duplicate_tenant():
    self.client.tenants.create(req_body['tenant']['name'], req_body['tenant']['description'], req_body['tenant']['enabled'])