from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_rule_not_found(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules', self.firewall_rule_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_rules'), json={'firewall_rules': []})])
    with mock.patch.object(self.cloud.network, 'delete_firewall_rule'), mock.patch.object(self.cloud.log, 'debug'):
        self.assertFalse(self.cloud.delete_firewall_rule(self.firewall_rule_name))
        self.cloud.network.delete_firewall_rule.assert_not_called()
        self.cloud.log.debug.assert_called_once()