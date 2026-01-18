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
def test_get_inherited_role_assignments_for_project_hierarchy(self):
    """Call ``GET /role_assignments?scope.OS-INHERIT:inherited_to``.

        Test Plan:

        - Create 2 roles
        - Create a hierarchy of projects with one root and one leaf project
        - Issue the URL to add a non-inherited user role to the root project
        - Issue the URL to add an inherited user role to the root project
        - Issue the URL to filter inherited to projects role assignments - this
          should return 1 role (inherited) on the root project.

        """
    root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
    non_inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_up_entity['links']['assignment'])
    inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
    self.put(inher_up_entity['links']['assignment'])
    collection_url = '/role_assignments?scope.OS-INHERIT:inherited_to=projects'
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
    self.assertRoleAssignmentInListResponse(r, inher_up_entity)
    non_inher_up_entity = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
    inher_up_entity['scope']['project']['id'] = leaf_id
    self.assertRoleAssignmentNotInListResponse(r, inher_up_entity)