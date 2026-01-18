import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_authenticate_and_get_roles_no_metadata(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    del user['id']
    new_user = PROVIDERS.identity_api.create_user(user)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    PROVIDERS.assignment_api.add_role_to_user_and_project(new_user['id'], self.project_baz['id'], role_member['id'])
    with self.make_request():
        user_ref = PROVIDERS.identity_api.authenticate(user_id=new_user['id'], password=user['password'])
    self.assertNotIn('password', user_ref)
    user.pop('password')
    self.assertLessEqual(user.items(), user_ref.items())
    role_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(new_user['id'], self.project_baz['id'])
    self.assertEqual(1, len(role_list))
    self.assertIn(role_member['id'], role_list)