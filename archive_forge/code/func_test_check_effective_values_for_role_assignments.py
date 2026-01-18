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
def test_check_effective_values_for_role_assignments(self):
    """Call ``GET & HEAD /role_assignments?effective=value``.

        Check the various ways of specifying the 'effective'
        query parameter.  If the 'effective' query parameter
        is included then this should always be treated as meaning 'True'
        unless it is specified as:

        {url}?effective=0

        This is by design to match the agreed way of handling
        policy checking on query/filter parameters.

        Test Plan:

        - Create two extra user for tests
        - Add these users to a group
        - Add a role assignment for the group on a domain
        - Get a list of all role assignments, checking one has been added
        - Then issue various request with different ways of defining
          the 'effective' query parameter. As we have tested the
          correctness of the data coming back when we get effective roles
          in other tests, here we just use the count of entities to
          know if we are getting effective roles or not

        """
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    PROVIDERS.identity_api.add_user_to_group(user1['id'], self.group['id'])
    PROVIDERS.identity_api.add_user_to_group(user2['id'], self.group['id'])
    collection_url = '/role_assignments'
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    existing_assignments = len(r.result.get('role_assignments'))
    gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=self.role_id)
    self.put(gd_entity['links']['assignment'])
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(r, gd_entity)
    collection_url = '/role_assignments?effective'
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
    collection_url = '/role_assignments?effective=0'
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
    collection_url = '/role_assignments?effective=False'
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
    collection_url = '/role_assignments?effective=True'
    r = self.get(collection_url, expected_status=http.client.OK)
    self.head(collection_url, expected_status=http.client.OK)
    self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)