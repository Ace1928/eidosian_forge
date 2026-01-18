from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_list_firewall_rules(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_rules'), json={'firewall_rules': [self.mock_firewall_rule]})])
    self.assertDictEqual(self.mock_firewall_rule, self.cloud.list_firewall_rules()[0])
    self.assert_calls()