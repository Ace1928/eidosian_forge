from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_user_with_grants(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user1 = unit.new_user_ref(domain_id=domain['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    group1 = unit.new_group_ref(domain_id=domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    group2 = unit.new_group_ref(domain_id=domain['id'])
    group2 = PROVIDERS.identity_api.create_group(group2)
    project1 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=self.role_admin['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group2['id'], project_id=project2['id'], role_id=self.role_admin['id'])
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertEqual(3, len(user_projects))