from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_group(self):
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_name), status_code=404), dict(method='GET', uri=self._make_mock_url('firewall_groups', name=self.firewall_group_name), json={'firewall_groups': [deepcopy(self.mock_returned_firewall_group)]}), dict(method='DELETE', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), status_code=204)])
    self.assertTrue(self.cloud.delete_firewall_group(self.firewall_group_name))
    self.assert_calls()