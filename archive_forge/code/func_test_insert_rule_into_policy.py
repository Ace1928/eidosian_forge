from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_insert_rule_into_policy(self):
    rule0 = FirewallRule(connection=self.cloud, **TestFirewallRule._mock_firewall_rule_attrs)
    _rule1_attrs = deepcopy(TestFirewallRule._mock_firewall_rule_attrs)
    _rule1_attrs.update(id='8068fc06-0e72-43f2-a76f-a51a33b46e08', name='after_rule')
    rule1 = FirewallRule(**_rule1_attrs)
    _rule2_attrs = deepcopy(TestFirewallRule._mock_firewall_rule_attrs)
    _rule2_attrs.update(id='c716382d-183b-475d-b500-dcc762f45ce3', name='before_rule')
    rule2 = FirewallRule(**_rule2_attrs)
    retrieved_policy = deepcopy(self.mock_firewall_policy)
    retrieved_policy['firewall_rules'] = [rule1['id'], rule2['id']]
    updated_policy = deepcopy(self.mock_firewall_policy)
    updated_policy['firewall_rules'] = [rule0['id'], rule1['id'], rule2['id']]
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_policies', name=self.firewall_policy_name), json={'firewall_policies': [retrieved_policy]}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule0['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule0['name']), json={'firewall_rules': [rule0]}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule1['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule1['name']), json={'firewall_rules': [rule1]}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule2['name']), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule2['name']), json={'firewall_rules': [rule2]}), dict(method='PUT', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id, 'insert_rule'), json=updated_policy, validate=dict(json={'firewall_rule_id': rule0['id'], 'insert_after': rule1['id'], 'insert_before': rule2['id']}))])
    r = self.cloud.insert_rule_into_policy(name_or_id=self.firewall_policy_name, rule_name_or_id=rule0['name'], insert_after=rule1['name'], insert_before=rule2['name'])
    self.assertDictEqual(updated_policy, r.to_dict())
    self.assert_calls()