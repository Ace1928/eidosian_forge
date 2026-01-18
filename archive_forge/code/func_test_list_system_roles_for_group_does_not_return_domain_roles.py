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
def test_list_system_roles_for_group_does_not_return_domain_roles(self):
    system_role_id = self._create_new_role()
    domain_role_id = self._create_new_role()
    group = self._create_group()
    domain_member_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'domain_id': group['domain_id'], 'group_id': group['id'], 'role_id': domain_role_id}
    self.put(domain_member_url)
    member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
    self.put(member_url)
    response = self.get('/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': group['domain_id'], 'group_id': group['id']})
    self.assertEqual(len(response.json_body['roles']), 1)
    collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
    response = self.get(collection_url)
    for role in response.json_body['roles']:
        self.assertNotEqual(role['id'], domain_role_id)
    response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
    self.assertValidRoleAssignmentListResponse(response, expected_length=1)