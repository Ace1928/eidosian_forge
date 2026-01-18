from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_groups(self):
    """Test retrieving projects for a list of groups.

        Test Plan:

        - Create two domains, four projects, seven groups and seven roles
        - Project1-3 are in Domain1, Project4 is in Domain2
        - Domain2/Project4 are spoilers
        - Project1 and 2 have direct group roles, Project3 has no direct
          roles but should inherit a group role from Domain1
        - Get the projects for the group roles that are assigned to Project1
          Project2 and the inherited one on Domain1. Depending on whether we
          have enabled inheritance, we should either get back just the projects
          with direct roles (Project 1 and 2) or also Project3 due to its
          inherited role from Domain1.

        """
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    project1 = PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain1['id'])
    project2 = PROVIDERS.resource_api.create_project(project2['id'], project2)
    project3 = unit.new_project_ref(domain_id=domain1['id'])
    project3 = PROVIDERS.resource_api.create_project(project3['id'], project3)
    project4 = unit.new_project_ref(domain_id=domain2['id'])
    project4 = PROVIDERS.resource_api.create_project(project4['id'], project4)
    group_list = []
    role_list = []
    for _ in range(7):
        group = unit.new_group_ref(domain_id=domain1['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_list.append(group)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], project_id=project1['id'], role_id=role_list[2]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[3]['id'], project_id=project2['id'], role_id=role_list[3]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[4]['id'], domain_id=domain2['id'], role_id=role_list[4]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[5]['id'], domain_id=domain2['id'], role_id=role_list[5]['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[6]['id'], project_id=project4['id'], role_id=role_list[6]['id'])
    group_id_list = [group_list[1]['id'], group_list[2]['id'], group_list[3]['id']]
    project_refs = PROVIDERS.assignment_api.list_projects_for_groups(group_id_list)
    self.assertThat(project_refs, matchers.HasLength(3))
    self.assertIn(project1, project_refs)
    self.assertIn(project2, project_refs)
    self.assertIn(project3, project_refs)