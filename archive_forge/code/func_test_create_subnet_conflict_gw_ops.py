import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_conflict_gw_ops(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', 'kooky']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=kooky']), json={'networks': [self.mock_network_rep]})])
    gateway = '192.168.200.3'
    self.assertRaises(exceptions.SDKException, self.cloud.create_subnet, 'kooky', self.subnet_cidr, gateway_ip=gateway, disable_gateway_ip=True)
    self.assert_calls()