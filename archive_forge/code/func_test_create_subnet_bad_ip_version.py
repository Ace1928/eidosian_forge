import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_bad_ip_version(self):
    """String ip_versions must be convertable to int"""
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', self.network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % self.network_name]), json={'networks': [self.mock_network_rep]})])
    with testtools.ExpectedException(exceptions.SDKException, 'ip_version must be an integer'):
        self.cloud.create_subnet(self.network_name, self.subnet_cidr, ip_version='4x')
    self.assert_calls()