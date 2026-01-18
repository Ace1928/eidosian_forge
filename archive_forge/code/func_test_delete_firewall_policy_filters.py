from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_delete_firewall_policy_filters(self):
    filters = {'project_id': self.mock_firewall_policy['project_id']}
    self.register_uris([dict(method='DELETE', uri=self._make_mock_url('firewall_policies', self.firewall_policy_id), json={}, status_code=204)])
    with mock.patch.object(self.cloud.network, 'find_firewall_policy', return_value=self.mock_firewall_policy), mock.patch.object(self.cloud.log, 'debug'):
        self.assertTrue(self.cloud.delete_firewall_policy(self.firewall_policy_name, filters))
        self.assert_calls()
        self.cloud.network.find_firewall_policy.assert_called_once_with(self.firewall_policy_name, ignore_missing=False, **filters)
        self.cloud.log.debug.assert_not_called()