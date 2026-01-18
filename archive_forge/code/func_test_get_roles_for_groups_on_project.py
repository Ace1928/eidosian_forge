from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_roles_for_groups_on_project(self):
    """Test retrieving group project roles.

        Test Plan:

        - Create two domains, two projects, six groups and six roles
        - Project1 is in Domain1, Project2 is in Domain2
        - Domain2/Project2 are spoilers
        - Assign a different direct group role to each project as well
          as both an inherited and non-inherited role to each domain
        - Get the group roles for Project 1 - depending on whether we have
          enabled inheritance, we should either get back just the direct role
          or both the direct one plus the inherited domain role from Domain 1

        """
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    project1 = unit.new_project_ref(domain_id=domain1['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain2['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    group_list = []
    group_id_list = []
    role_list = []
    for _ in range(6):
        group = unit.new_group_ref(domain_id=domain1['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_list.append(group)
        group_id_list.append(group['id'])
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], project_id=project1['id'], role_id=role_list[2]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[3]['id'], domain_id=domain2['id'], role_id=role_list[3]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[4]['id'], domain_id=domain2['id'], role_id=role_list[4]['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[5]['id'], project_id=project2['id'], role_id=role_list[5]['id'])
    role_refs = PROVIDERS.assignment_api.get_roles_for_groups(group_id_list, project_id=project1['id'])
    self.assertThat(role_refs, matchers.HasLength(2))
    self.assertIn(role_list[1], role_refs)
    self.assertIn(role_list[2], role_refs)