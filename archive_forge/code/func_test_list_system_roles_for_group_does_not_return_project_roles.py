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
def test_list_system_roles_for_group_does_not_return_project_roles(self):
    system_role_id = self._create_new_role()
    project_role_id = self._create_new_role()
    group = self._create_group()
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': project_role_id}
    self.put(member_url)
    collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
    response = self.get(collection_url)
    for role in response.json_body['roles']:
        self.assertNotEqual(role['id'], project_role_id)
    response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)