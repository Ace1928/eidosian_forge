import copy
from openstack import exceptions
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.tests.unit import base
def test_create_qos_dscp_marking_rule(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies'], qs_elements=['name=%s' % self.policy_name]), json={'policies': [self.mock_policy]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_id, 'dscp_marking_rules']), json={'dscp_marking_rule': self.mock_rule})])
    rule = self.cloud.create_qos_dscp_marking_rule(self.policy_name, dscp_mark=self.rule_dscp_mark)
    self._compare_rules(self.mock_rule, rule)
    self.assert_calls()