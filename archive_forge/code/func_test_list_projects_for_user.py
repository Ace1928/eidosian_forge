from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_user(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user1 = unit.new_user_ref(domain_id=domain['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertEqual(0, len(user_projects))
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_baz['id'], role_id=self.role_member['id'])
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertEqual(2, len(user_projects))