import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_cannot_update_immutable_role_while_unsetting_immutable(self):
    role = unit.new_role_ref()
    role_id = role['id']
    role['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.role_api.create_role(role_id, role)
    update_role = {'name': uuid.uuid4().hex, 'options': {ro_opt.IMMUTABLE_OPT.option_name: True}}
    self.assertRaises(exception.ResourceUpdateForbidden, PROVIDERS.role_api.update_role, role_id, update_role)