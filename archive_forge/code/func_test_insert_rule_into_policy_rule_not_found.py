from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_insert_rule_into_policy_rule_not_found(self):
    rule_name = 'unknown_rule'
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={'firewall_policy': self.mock_firewall_policy}), dict(method='GET', uri=self._make_mock_url('firewall_rules', rule_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules', name=rule_name), json={'firewall_rules': []})])
    self.assertRaises(exceptions.ResourceNotFound, self.cloud.insert_rule_into_policy, self.firewall_policy_id, rule_name)
    self.assert_calls()