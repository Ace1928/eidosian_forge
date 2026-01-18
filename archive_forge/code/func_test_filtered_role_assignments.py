import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_filtered_role_assignments(self):
    """Call ``GET /role_assignments?filters``.

        Test Plan:

        - Create extra users, group, role and project for tests
        - Make the following assignments:
          Give group1, role1 on project1 and domain
          Give user1, role2 on project1 and domain
          Make User1 a member of Group1
        - Test a series of single filter list calls, checking that
          the correct results are obtained
        - Test a multi-filtered list call
        - Test listing all effective roles for a given user
        - Test the equivalent of the list of roles in a project scoped
          token (all effective roles for a user on a project)

        """
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    group1 = unit.new_group_ref(domain_id=self.domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], group1['id'])
    project1 = unit.new_project_ref(domain_id=self.domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    self.role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(self.role1['id'], self.role1)
    self.role2 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(self.role2['id'], self.role2)
    gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=group1['id'], role_id=self.role1['id'])
    self.put(gd_entity['links']['assignment'])
    ud_entity = self.build_role_assignment_entity(domain_id=self.domain_id, user_id=user1['id'], role_id=self.role2['id'])
    self.put(ud_entity['links']['assignment'])
    gp_entity = self.build_role_assignment_entity(project_id=project1['id'], group_id=group1['id'], role_id=self.role1['id'])
    self.put(gp_entity['links']['assignment'])
    up_entity = self.build_role_assignment_entity(project_id=project1['id'], user_id=user1['id'], role_id=self.role2['id'])
    self.put(up_entity['links']['assignment'])
    gs_entity = self.build_role_assignment_entity(system='all', group_id=group1['id'], role_id=self.role1['id'])
    self.put(gs_entity['links']['assignment'])
    us_entity = self.build_role_assignment_entity(system='all', user_id=user1['id'], role_id=self.role2['id'])
    self.put(us_entity['links']['assignment'])
    us2_entity = self.build_role_assignment_entity(system='all', user_id=user2['id'], role_id=self.role2['id'])
    self.put(us2_entity['links']['assignment'])
    collection_url = '/role_assignments?scope.project.id=%s' % project1['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    self.assertRoleAssignmentInListResponse(r, gp_entity)
    collection_url = '/role_assignments?scope.domain.id=%s' % self.domain['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    self.assertRoleAssignmentInListResponse(r, gd_entity)
    collection_url = '/role_assignments?user.id=%s' % user1['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    collection_url = '/role_assignments?group.id=%s' % group1['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, gd_entity)
    self.assertRoleAssignmentInListResponse(r, gp_entity)
    collection_url = '/role_assignments?role.id=%s' % self.role1['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, gd_entity)
    self.assertRoleAssignmentInListResponse(r, gp_entity)
    self.assertRoleAssignmentInListResponse(r, gs_entity)
    collection_url = '/role_assignments?role.id=%s' % self.role2['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=4, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    self.assertRoleAssignmentInListResponse(r, us_entity)
    collection_url = '/role_assignments?user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    collection_url = '/role_assignments?effective&user.id=%s' % user1['id']
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=4, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    gp1_link = self.build_role_assignment_link(project_id=project1['id'], group_id=group1['id'], role_id=self.role1['id'])
    gd1_link = self.build_role_assignment_link(domain_id=self.domain_id, group_id=group1['id'], role_id=self.role1['id'])
    up1_entity = self.build_role_assignment_entity(link=gp1_link, project_id=project1['id'], user_id=user1['id'], role_id=self.role1['id'])
    ud1_entity = self.build_role_assignment_entity(link=gd1_link, domain_id=self.domain_id, user_id=user1['id'], role_id=self.role1['id'])
    self.assertRoleAssignmentInListResponse(r, up1_entity)
    self.assertRoleAssignmentInListResponse(r, ud1_entity)
    collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, up_entity)
    self.assertRoleAssignmentInListResponse(r, up1_entity)