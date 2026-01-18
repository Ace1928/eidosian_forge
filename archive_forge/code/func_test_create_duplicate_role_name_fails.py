import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_create_duplicate_role_name_fails(self):
    role_id = uuid.uuid4().hex
    role = unit.new_role_ref(id=role_id, name='fake1name')
    PROVIDERS.role_api.create_role(role_id, role)
    new_role_id = uuid.uuid4().hex
    role['id'] = new_role_id
    self.assertRaises(exception.Conflict, PROVIDERS.role_api.create_role, new_role_id, role)