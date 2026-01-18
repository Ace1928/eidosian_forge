from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_role_check_role_grant(self):
    role = unit.new_role_ref()
    alt_role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    PROVIDERS.role_api.create_role(alt_role['id'], alt_role)
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], role['id'])
    PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], alt_role['id'])
    PROVIDERS.role_api.delete_role(role['id'])
    roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
    self.assertNotIn(role['id'], roles_ref)
    self.assertIn(alt_role['id'], roles_ref)