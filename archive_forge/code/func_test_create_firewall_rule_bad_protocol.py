from copy import deepcopy
from unittest import mock
from openstack import exceptions
from openstack.network.v2.firewall_group import FirewallGroup
from openstack.network.v2.firewall_policy import FirewallPolicy
from openstack.network.v2.firewall_rule import FirewallRule
from openstack.tests.unit import base
def test_create_firewall_rule_bad_protocol(self):
    bad_rule = self._mock_firewall_rule_attrs.copy()
    del bad_rule['id']
    bad_rule['ip_version'] = 5
    self.register_uris([dict(method='POST', uri=self._make_mock_url('firewall_rules'), status_code=400, json={})])
    self.assertRaises(exceptions.BadRequestException, self.cloud.create_firewall_rule, **bad_rule)
    self.assert_calls()