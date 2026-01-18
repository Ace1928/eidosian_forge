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
def test_list_system_roles_for_group_returns_none_without_assignment(self):
    group = self._create_group()
    collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
    response = self.get(collection_url)
    self.assertEqual(response.json_body['roles'], [])
    response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=0)