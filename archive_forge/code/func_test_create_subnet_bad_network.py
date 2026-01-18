import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_bad_network(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', 'duck']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=duck']), json={'networks': [self.mock_network_rep]})])
    self.assertRaises(exceptions.SDKException, self.cloud.create_subnet, 'duck', self.subnet_cidr)
    self.assert_calls()