import copy
from openstack import exceptions
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.tests.unit import base
def test_create_qos_dscp_marking_rule_no_qos_extension(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': []})])
    self.assertRaises(exceptions.SDKException, self.cloud.create_qos_dscp_marking_rule, self.policy_name, dscp_mark=16)
    self.assert_calls()