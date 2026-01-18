import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_add_user_to_group_returns_not_found(self):
    domain = self._get_domain_fixture()
    new_user = unit.new_user_ref(domain_id=domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.add_user_to_group, new_user['id'], uuid.uuid4().hex)
    new_group = unit.new_group_ref(domain_id=domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.add_user_to_group, uuid.uuid4().hex, new_group['id'])
    self.assertRaises(exception.NotFound, PROVIDERS.identity_api.add_user_to_group, uuid.uuid4().hex, uuid.uuid4().hex)