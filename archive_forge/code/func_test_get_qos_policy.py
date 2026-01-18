import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_get_qos_policy(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', self.policy_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies'], qs_elements=['name=%s' % self.policy_name]), json={'policies': [self.mock_policy]})])
    r = self.cloud.get_qos_policy(self.policy_name)
    self.assertIsNotNone(r)
    self._compare_policies(self.mock_policy, r)
    self.assert_calls()