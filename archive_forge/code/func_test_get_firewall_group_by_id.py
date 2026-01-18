from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_get_firewall_group_by_id(self):
    returned_group = deepcopy(self.mock_returned_firewall_group)
    self.register_uris([dict(method='GET', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), json={'firewall_group': returned_group})])
    r = self.cloud.get_firewall_group(self.firewall_group_id)
    self.assertDictEqual(returned_group, r.to_dict())
    self.assert_calls()