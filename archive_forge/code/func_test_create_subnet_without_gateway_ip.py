import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_without_gateway_ip(self):
    pool = [{'start': '192.168.199.2', 'end': '192.168.199.254'}]
    dns = ['8.8.8.8']
    mock_subnet_rep = copy.copy(self.mock_subnet_rep)
    mock_subnet_rep['allocation_pools'] = pool
    mock_subnet_rep['dns_nameservers'] = dns
    mock_subnet_rep['gateway_ip'] = None
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', self.network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % self.network_name]), json={'networks': [self.mock_network_rep]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnet': mock_subnet_rep}, validate=dict(json={'subnet': {'cidr': self.subnet_cidr, 'enable_dhcp': False, 'ip_version': 4, 'network_id': self.mock_network_rep['id'], 'allocation_pools': pool, 'gateway_ip': None, 'dns_nameservers': dns}}))])
    subnet = self.cloud.create_subnet(self.network_name, self.subnet_cidr, allocation_pools=pool, dns_nameservers=dns, disable_gateway_ip=True)
    self._compare_subnets(mock_subnet_rep, subnet)
    self.assert_calls()