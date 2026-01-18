from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_user_with_inherited_group_grants(self):
    """Test inherited group roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create two domains, each with two projects
        - Create a user and group
        - Make the user a member of the group
        - Assign a user role two projects, an inherited
          group role to one domain and an inherited regular role on
          the other domain
        - Get a list of projects for user, should return both pairs of projects
          from the domain, plus the one separate project

        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    project1 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    project3 = unit.new_project_ref(domain_id=domain2['id'])
    PROVIDERS.resource_api.create_project(project3['id'], project3)
    project4 = unit.new_project_ref(domain_id=domain2['id'])
    PROVIDERS.resource_api.create_project(project4['id'], project4)
    user1 = unit.new_user_ref(domain_id=domain['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    group1 = unit.new_group_ref(domain_id=domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project3['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertEqual(5, len(user_projects))
    test_plan = {'entities': {'domains': [{'projects': 1}, {'users': 1, 'groups': 1, 'projects': 2}, {'projects': 2}], 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 0, 'project': 3}, {'user': 0, 'role': 1, 'domain': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 1, 'domain': 2, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 3}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'domain': 2}}, {'user': 0, 'role': 1, 'project': 4, 'indirect': {'domain': 2}}]}]}
    self.execute_assignment_plan(test_plan)