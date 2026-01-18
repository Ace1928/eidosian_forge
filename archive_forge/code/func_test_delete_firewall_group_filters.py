from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_group_filters(self):
    filters = {'project_id': self.mock_firewall_group['project_id']}
    self.register_uris([dict(method='DELETE', uri=self._make_mock_url('firewall_groups', self.firewall_group_id), status_code=204)])
    with mock.patch.object(self.cloud.network, 'find_firewall_group', return_value=deepcopy(self.mock_firewall_group)):
        self.assertTrue(self.cloud.delete_firewall_group(self.firewall_group_name, filters))
        self.assert_calls()
        self.cloud.network.find_firewall_group.assert_called_once_with(self.firewall_group_name, ignore_missing=False, **filters)