import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_delete_subnet_multiple_found(self):
    subnet1 = dict(id='123', name=self.subnet_name)
    subnet2 = dict(id='456', name=self.subnet_name)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', self.subnet_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets'], qs_elements=['name=%s' % self.subnet_name]), json={'subnets': [subnet1, subnet2]})])
    self.assertRaises(exceptions.SDKException, self.cloud.delete_subnet, self.subnet_name)
    self.assert_calls()