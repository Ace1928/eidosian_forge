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
def test_assign_system_role_to_user(self):
    system_role_id = self._create_new_role()
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.put(member_url)
    self.head(member_url)
    collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
    roles = self.get(collection_url).json_body['roles']
    self.assertEqual(len(roles), 1)
    self.assertEqual(roles[0]['id'], system_role_id)
    self.head(collection_url, expected_status=http.client.OK)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertValidRoleAssignmentListResponse(response)