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
def test_get_role_assignments_for_project_tree(self):
    """Get role_assignment?scope.project.id=X&include_subtree``.

        Test Plan:

        - Create 2 roles and a hierarchy of projects with one root and one leaf
        - Issue the URL to add a non-inherited user role to the root project
          and the leaf project
        - Issue the URL to get role assignments for the root project but
          not the subtree - this should return just the root assignment
        - Issue the URL to get role assignments for the root project and
          it's subtree - this should return both assignments
        - Check that explicitly setting include_subtree to False is the
          equivalent to not including it at all in the query.

        """
    root_id, leaf_id, non_inherited_role_id, unused_role_id = self._setup_hierarchical_projects_scenario()
    non_inher_entity_root = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_entity_root['links']['assignment'])
    non_inher_entity_leaf = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_entity_leaf['links']['assignment'])
    collection_url = '/role_assignments?scope.project.id=%(project)s' % {'project': root_id}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    self.assertThat(r.result['role_assignments'], matchers.HasLength(1))
    self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)
    collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=True' % {'project': root_id}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    self.assertThat(r.result['role_assignments'], matchers.HasLength(2))
    self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)
    self.assertRoleAssignmentInListResponse(r, non_inher_entity_leaf)
    collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=0' % {'project': root_id}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    self.assertThat(r.result['role_assignments'], matchers.HasLength(1))
    self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)