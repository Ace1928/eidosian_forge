from openstack import exceptions
from openstack.tests.functional import base
def test_create_qos_policy_default(self):
    if not self.operator_cloud._has_neutron_extension('qos-default'):
        self.skipTest("'qos-default' network extension not supported by cloud")
    policy = self.operator_cloud.create_qos_policy(name=self.policy_name, default=True)
    self.assertIn('id', policy)
    self.assertEqual(self.policy_name, policy['name'])
    self.assertFalse(policy['is_shared'])
    self.assertTrue(policy['is_default'])