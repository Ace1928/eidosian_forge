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
def test_list_system_roles_for_user_does_not_return_domain_roles(self):
    system_role_id = self._create_new_role()
    domain_role_id = self._create_new_role()
    domain_member_url = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id'], 'role_id': domain_role_id}
    self.put(domain_member_url)
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.put(member_url)
    response = self.get('/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['roles']), 1)
    collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
    response = self.get(collection_url)
    for role in response.json_body['roles']:
        self.assertNotEqual(role['id'], domain_role_id)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['role_assignments']), 1)
    system_assignment = response.json_body['role_assignments'][0]
    self.assertEqual(system_assignment['role']['id'], system_role_id)
    self.assertTrue(system_assignment['scope']['system']['all'])
    path = '/role_assignments?scope.domain.id=%(domain_id)s&user.id=%(user_id)s' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id']}
    response = self.get(path)
    self.assertEqual(len(response.json_body['role_assignments']), 1)
    domain_assignment = response.json_body['role_assignments'][0]
    self.assertEqual(domain_assignment['role']['id'], domain_role_id)