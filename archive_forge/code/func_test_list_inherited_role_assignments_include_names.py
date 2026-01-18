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
def test_list_inherited_role_assignments_include_names(self):
    """Call ``GET /role_assignments?include_names``.

        Test goal: ensure calling list role assignments including names
        honors the inherited role assignments flag.

        Test plan:
        - Create a role and a domain with a user;
        - Create a inherited role assignment;
        - List role assignments for that user;
        - List role assignments for that user including names.

        """
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    assignment = self.build_role_assignment_entity(role_id=role['id'], domain_id=domain['id'], user_id=user['id'], inherited_to_projects=True)
    assignment_names = self.build_role_assignment_entity_include_names(role_ref=role, domain_ref=domain, user_ref=user, inherited_assignment=True)
    self.assertEqual('projects', assignment['scope']['OS-INHERIT:inherited_to'])
    self.assertEqual('projects', assignment_names['scope']['OS-INHERIT:inherited_to'])
    self.assertEqual(assignment['links']['assignment'], assignment_names['links']['assignment'])
    self.put(assignment['links']['assignment'])
    collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
    result = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(result, assignment)
    collection_url = '/role_assignments?include_names&user.id=%(user_id)s' % {'user_id': user['id']}
    result = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(result, assignment_names)