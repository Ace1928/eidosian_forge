import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_roles(self):
    roles = PROVIDERS.role_api.list_roles()
    self.assertEqual(len(default_fixtures.ROLES), len(roles))
    role_ids = set((role['id'] for role in roles))
    expected_role_ids = set((role['id'] for role in default_fixtures.ROLES))
    self.assertEqual(expected_role_ids, role_ids)