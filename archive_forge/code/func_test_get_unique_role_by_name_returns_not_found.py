import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_unique_role_by_name_returns_not_found(self):
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_unique_role_by_name, uuid.uuid4().hex)