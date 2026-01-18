from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_user_with_project_roles(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_bar['id'], self.role_member['id'])
    PROVIDERS.identity_api.delete_user(user['id'])
    self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, user['id'])