import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_delete_subnet_using_id(self):
    subnet1 = dict(id='123', name=self.subnet_name)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', subnet1['id']]), json=subnet1), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', subnet1['id']]), json={})])
    self.assertTrue(self.cloud.delete_subnet(subnet1['id']))
    self.assert_calls()