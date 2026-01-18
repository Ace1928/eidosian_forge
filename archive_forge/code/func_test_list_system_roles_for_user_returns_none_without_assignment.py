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
def test_list_system_roles_for_user_returns_none_without_assignment(self):
    collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
    response = self.get(collection_url)
    self.assertEqual(response.json_body['roles'], [])
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['role_assignments']), 0)
    self.assertValidRoleAssignmentListResponse(response)