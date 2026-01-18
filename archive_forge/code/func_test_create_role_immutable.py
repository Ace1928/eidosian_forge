import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_create_role_immutable(self):
    role = unit.new_role_ref()
    role_id = role['id']
    role['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    role_created = PROVIDERS.role_api.create_role(role_id, role)
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue('options' in role_created)
    self.assertTrue('options' in role_via_manager)
    self.assertTrue(role_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    self.assertTrue(role_created['options'][ro_opt.IMMUTABLE_OPT.option_name])