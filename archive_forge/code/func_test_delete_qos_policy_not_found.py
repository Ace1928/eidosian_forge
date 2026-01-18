import copy
from openstack import exceptions
from openstack.network.v2 import qos_policy as _policy
from openstack.tests.unit import base
def test_delete_qos_policy_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies', 'goofy']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'qos', 'policies'], qs_elements=['name=goofy']), json={'policies': []})])
    self.assertFalse(self.cloud.delete_qos_policy('goofy'))
    self.assert_calls()