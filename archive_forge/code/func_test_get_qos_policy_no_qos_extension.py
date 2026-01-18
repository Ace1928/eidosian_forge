import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_get_qos_policy_no_qos_extension(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': []})])
    self.assertRaises(exceptions.SDKException, self.cloud.get_qos_policy, self.policy_name)
    self.assert_calls()