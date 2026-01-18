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
def test_get_effective_role_assignments(self):
    """Call ``GET /role_assignments?effective``.

        Test Plan:

        - Create two extra user for tests
        - Add these users to a group
        - Add a role assignment for the group on a domain
        - Get a list of all role assignments, checking one has been added
        - Then get a list of all effective role assignments - the group
          assignment should have turned into assignments on the domain
          for each of the group members.

        """
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    PROVIDERS.identity_api.add_user_to_group(user1['id'], self.group['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], self.group['id'])
    collection_url = '/role_assignments'
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    existing_assignments = len(r.result.get('role_assignments'))
    gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=self.role_id)
    self.put(gd_entity['links']['assignment'])
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, gd_entity)
    collection_url = '/role_assignments?effective'
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
    ud_entity = self.build_role_assignment_entity(link=gd_entity['links']['assignment'], domain_id=self.domain_id, user_id=user1['id'], role_id=self.role_id)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    ud_entity = self.build_role_assignment_entity(link=gd_entity['links']['assignment'], domain_id=self.domain_id, user_id=user2['id'], role_id=self.role_id)
    self.assertRoleAssignmentInListResponse(r, ud_entity)