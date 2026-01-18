import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_delete_subnet_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', 'goofy']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets'], qs_elements=['name=goofy']), json={'subnets': []})])
    self.assertFalse(self.cloud.delete_subnet('goofy'))
    self.assert_calls()