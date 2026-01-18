import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_cannot_delete_immutable_role(self):
    role = unit.new_role_ref()
    role_id = role['id']
    role['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.role_api.create_role(role_id, role)
    self.assertRaises(exception.ResourceDeleteForbidden, PROVIDERS.role_api.delete_role, role_id)