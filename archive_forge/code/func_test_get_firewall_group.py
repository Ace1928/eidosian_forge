from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_get_firewall_group(self):
    returned_group = deepcopy(self.mock_returned_firewall_group)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_groups', name=self.firewall_group_name), json={'firewall_groups': [returned_group]})])
    self.assertDictEqual(returned_group, self.cloud.get_firewall_group(self.firewall_group_name))
    self.assert_calls()