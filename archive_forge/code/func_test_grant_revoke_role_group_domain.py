import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def test_grant_revoke_role_group_domain(self):
    if self.identity_version in ('2', '2.0'):
        self.skipTest('Identity service does not support domain or group')
    role_name = self.role_prefix + '_grant_group_domain'
    role = self.operator_cloud.create_role(role_name)
    group_name = self.group_prefix + '_group_domain'
    group = self.operator_cloud.create_group(name=group_name, description='test group', domain='default')
    self.assertTrue(self.operator_cloud.grant_role(role_name, group=group['id'], domain='default'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'group': group['id'], 'domain': self.operator_cloud.get_domain('default')['id']})
    self.assertIsInstance(assignments, list)
    self.assertEqual(1, len(assignments))
    self.assertTrue(self.operator_cloud.revoke_role(role_name, group=group['id'], domain='default'))
    assignments = self.operator_cloud.list_role_assignments({'role': role['id'], 'group': group['id'], 'domain': self.operator_cloud.get_domain('default')['id']})
    self.assertIsInstance(assignments, list)
    self.assertEqual(0, len(assignments))