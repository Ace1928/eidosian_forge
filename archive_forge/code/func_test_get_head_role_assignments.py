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
def test_get_head_role_assignments(self):
    """Call ``GET & HEAD /role_assignments``.

        The sample data set up already has a user, group and project
        that is part of self.domain. We use these plus a new user
        we create as our data set, making sure we ignore any
        role assignments that are already in existence.

        Since we don't yet support a first class entity for role
        assignments, we are only testing the LIST API.  To create
        and delete the role assignments we use the old grant APIs.

        Test Plan:

        - Create extra user for tests
        - Get a list of all existing role assignments
        - Add a new assignment for each of the four combinations, i.e.
          group+domain, user+domain, group+project, user+project, using
          the same role each time
        - Get a new list of all role assignments, checking these four new
          ones have been added
        - Then delete the four we added
        - Get a new list of all role assignments, checking the four have
          been removed

        """
    time = datetime.datetime.utcnow()
    with freezegun.freeze_time(time) as frozen_datetime:
        user1 = unit.new_user_ref(domain_id=self.domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        collection_url = '/role_assignments'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.head(collection_url, expected_status=http.client.OK)
        existing_assignments = len(r.result.get('role_assignments'))
        gd_entity = self.build_role_assignment_entity(domain_id=self.domain_id, group_id=self.group_id, role_id=role['id'])
        self.put(gd_entity['links']['assignment'])
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        self.head(collection_url, expected_status=http.client.OK)
        ud_entity = self.build_role_assignment_entity(domain_id=self.domain_id, user_id=user1['id'], role_id=role['id'])
        self.put(ud_entity['links']['assignment'])
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 2, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        self.head(collection_url, expected_status=http.client.OK)
        gp_entity = self.build_role_assignment_entity(project_id=self.project_id, group_id=self.group_id, role_id=role['id'])
        self.put(gp_entity['links']['assignment'])
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 3, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, gp_entity)
        self.head(collection_url, expected_status=http.client.OK)
        up_entity = self.build_role_assignment_entity(project_id=self.project_id, user_id=user1['id'], role_id=role['id'])
        self.put(up_entity['links']['assignment'])
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments + 4, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, up_entity)
        self.head(collection_url, expected_status=http.client.OK)
        self.delete(gd_entity['links']['assignment'])
        self.delete(ud_entity['links']['assignment'])
        self.delete(gp_entity['links']['assignment'])
        self.delete(up_entity['links']['assignment'])
        frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=existing_assignments, resource_url=collection_url)
        self.assertRoleAssignmentNotInListResponse(r, gd_entity)
        self.assertRoleAssignmentNotInListResponse(r, ud_entity)
        self.assertRoleAssignmentNotInListResponse(r, gp_entity)
        self.assertRoleAssignmentNotInListResponse(r, up_entity)
        self.head(collection_url, expected_status=http.client.OK)