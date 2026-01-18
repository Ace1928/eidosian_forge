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
def test_list_system_roles_for_user_does_not_return_project_roles(self):
    system_role_id = self._create_new_role()
    member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
    self.put(member_url)
    response = self.get('/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project['id'], 'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['roles']), 1)
    project_role_id = response.json_body['roles'][0]['id']
    collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
    response = self.get(collection_url)
    for role in response.json_body['roles']:
        self.assertNotEqual(role['id'], project_role_id)
    response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
    self.assertEqual(len(response.json_body['role_assignments']), 1)
    system_assignment = response.json_body['role_assignments'][0]
    self.assertEqual(system_assignment['role']['id'], system_role_id)
    self.assertTrue(system_assignment['scope']['system']['all'])
    path = '/role_assignments?scope.project.id=%(project_id)s&user.id=%(user_id)s' % {'project_id': self.project['id'], 'user_id': self.user['id']}
    response = self.get(path)
    self.assertEqual(len(response.json_body['role_assignments']), 1)
    project_assignment = response.json_body['role_assignments'][0]
    self.assertEqual(project_assignment['role']['id'], project_role_id)