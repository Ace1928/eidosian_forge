import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_string_ip_version(self):
    """Allow ip_version as a string"""
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', self.network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % self.network_name]), json={'networks': [self.mock_network_rep]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnet': self.mock_subnet_rep}, validate=dict(json={'subnet': {'cidr': self.subnet_cidr, 'enable_dhcp': False, 'ip_version': 4, 'network_id': self.mock_network_rep['id']}}))])
    subnet = self.cloud.create_subnet(self.network_name, self.subnet_cidr, ip_version='4')
    self._compare_subnets(self.mock_subnet_rep, subnet)
    self.assert_calls()