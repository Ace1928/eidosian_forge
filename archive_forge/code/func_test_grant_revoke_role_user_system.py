import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_grant_revoke_role_user_system(self):
    role_name = self.role_prefix + '_grant_user_system'
    role = self.operator_cloud.create_role(role_name)
    user_name = self.user_prefix + '_user_system'
    user_email = 'nobody@nowhere.com'
    user = self._create_user(name=user_name, email=user_email, default_project='demo')
    self.assertTrue(self.operator_cloud.grant_role(role_name, user=user['id'], system='all'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'user': user['id'], 'system': 'all'})
    self.assertIsInstance(assignments, list)
    self.assertEqual(1, len(assignments))
    self.assertTrue(self.operator_cloud.revoke_role(role_name, user=user['id'], system='all'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'user': user['id'], 'system': 'all'})
    self.assertIsInstance(assignments, list)
    self.assertEqual(0, len(assignments))