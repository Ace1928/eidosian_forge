import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_role_returns_not_found(self):
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, uuid.uuid4().hex)