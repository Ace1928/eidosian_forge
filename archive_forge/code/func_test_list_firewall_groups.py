from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_list_firewall_groups(self):
    returned_attrs = deepcopy(self.mock_returned_firewall_group)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups'), json={'firewall_groups': [returned_attrs, returned_attrs]})])
    group = FirewallGroup(connection=self.cloud, **returned_attrs)
    self.assertListEqual([group, group], self.cloud.list_firewall_groups())
    self.assert_calls()