from openstack import exceptions
from openstack.tests.functional import base
def test_create_qos_bandwidth_limit_rule_direction(self):
    if not self.operator_cloud._has_neutron_extension('qos-bw-limit-direction'):
        self.skipTest("'qos-bw-limit-direction' network extension not supported by cloud")
    max_kbps = 1500
    direction = 'ingress'
    updated_direction = 'egress'
    rule = self.operator_cloud.create_qos_bandwidth_limit_rule(self.policy['id'], max_kbps=max_kbps, direction=direction)
    self.assertIn('id', rule)
    self.assertEqual(max_kbps, rule['max_kbps'])
    self.assertEqual(direction, rule['direction'])
    updated_rule = self.operator_cloud.update_qos_bandwidth_limit_rule(self.policy['id'], rule['id'], direction=updated_direction)
    self.assertIn('id', updated_rule)
    self.assertEqual(max_kbps, updated_rule['max_kbps'])
    self.assertEqual(updated_direction, updated_rule['direction'])