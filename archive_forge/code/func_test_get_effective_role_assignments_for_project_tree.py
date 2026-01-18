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
def test_get_effective_role_assignments_for_project_tree(self):
    """Get role_assignment ?project_id=X&include_subtree=True&effective``.

        Test Plan:

        - Create 2 roles and a hierarchy of projects with one root and 4 levels
          of child project
        - Issue the URL to add a non-inherited user role to the root project
          and a level 1 project
        - Issue the URL to add an inherited user role on the level 2 project
        - Issue the URL to get effective role assignments for the level 1
          project and it's subtree - this should return a role (non-inherited)
          on the level 1 project and roles (inherited) on each of the level
          2, 3 and 4 projects

        """
    root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
    level2 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=leaf_id)
    level3 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=level2['id'])
    level4 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=level3['id'])
    PROVIDERS.resource_api.create_project(level2['id'], level2)
    PROVIDERS.resource_api.create_project(level3['id'], level3)
    PROVIDERS.resource_api.create_project(level4['id'], level4)
    non_inher_entity_root = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_entity_root['links']['assignment'])
    non_inher_entity_leaf = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_entity_leaf['links']['assignment'])
    inher_entity = self.build_role_assignment_entity(project_id=level2['id'], user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
    self.put(inher_entity['links']['assignment'])
    collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=True&effective' % {'project': leaf_id}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
    self.assertThat(r.result['role_assignments'], matchers.HasLength(3))
    self.assertRoleAssignmentNotInListResponse(r, non_inher_entity_root)
    self.assertRoleAssignmentInListResponse(r, non_inher_entity_leaf)
    inher_entity['scope']['project']['id'] = level3['id']
    self.assertRoleAssignmentInListResponse(r, inher_entity)
    inher_entity['scope']['project']['id'] = level4['id']
    self.assertRoleAssignmentInListResponse(r, inher_entity)