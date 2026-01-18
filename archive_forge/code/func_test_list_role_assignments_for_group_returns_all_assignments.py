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
def test_list_role_assignments_for_group_returns_all_assignments(self):
    system_role_id = self._create_new_role()
    group = self._create_group()
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    response = self.get('/role_assignments?group.id=%(group_id)s' % {'group_id': group['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=2)