from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_policy(self):
    passed_attrs = deepcopy(self._mock_firewall_policy_attrs)
    del passed_attrs['id']
    created_attrs = deepcopy(self._mock_firewall_policy_attrs)
    created_attrs['firewall_rules'][0] = TestFirewallRule.firewall_rule_id
    created_policy = FirewallPolicy(connection=self.cloud, **created_attrs)
    validate_attrs = deepcopy(created_attrs)
    del validate_attrs['id']
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', TestFirewallRule.firewall_rule_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=TestFirewallRule.firewall_rule_name), json={'firewall_rules': [TestFirewallRule._mock_firewall_rule_attrs]}), dict(method='POST', uri=self._make_mock_url('firewall_policies'), json={'firewall_policy': created_attrs}, validate=dict(json={'firewall_policy': validate_attrs}))])
    res = self.cloud.create_firewall_policy(**passed_attrs)
    self.assertDictEqual(created_policy, res.to_dict())
    self.assert_calls()