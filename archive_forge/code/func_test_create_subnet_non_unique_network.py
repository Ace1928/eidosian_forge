import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_non_unique_network(self):
    net1 = dict(id='123', name=self.network_name)
    net2 = dict(id='456', name=self.network_name)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', self.network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % self.network_name]), json={'networks': [net1, net2]})])
    self.assertRaises(exceptions.SDKException, self.cloud.create_subnet, self.network_name, self.subnet_cidr)
    self.assert_calls()