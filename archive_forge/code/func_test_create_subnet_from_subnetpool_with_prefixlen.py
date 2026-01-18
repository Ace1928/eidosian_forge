import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_create_subnet_from_subnetpool_with_prefixlen(self):
    pool = [{'start': '172.16.0.2', 'end': '172.16.0.15'}]
    id = '143296eb-7f47-4755-835c-488123475604'
    gateway = '172.16.0.1'
    dns = ['8.8.8.8']
    routes = [{'destination': '0.0.0.0/0', 'nexthop': '123.456.78.9'}]
    mock_subnet_rep = copy.copy(self.mock_subnet_rep)
    mock_subnet_rep['allocation_pools'] = pool
    mock_subnet_rep['dns_nameservers'] = dns
    mock_subnet_rep['host_routes'] = routes
    mock_subnet_rep['gateway_ip'] = gateway
    mock_subnet_rep['subnetpool_id'] = self.mock_subnetpool_rep['id']
    mock_subnet_rep['cidr'] = self.subnetpool_cidr
    mock_subnet_rep['id'] = id
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', self.network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % self.network_name]), json={'networks': [self.mock_network_rep]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnet': mock_subnet_rep}, validate=dict(json={'subnet': {'enable_dhcp': False, 'ip_version': 4, 'network_id': self.mock_network_rep['id'], 'allocation_pools': pool, 'dns_nameservers': dns, 'use_default_subnetpool': True, 'prefixlen': self.prefix_length, 'host_routes': routes}}))])
    subnet = self.cloud.create_subnet(self.network_name, allocation_pools=pool, dns_nameservers=dns, use_default_subnetpool=True, prefixlen=self.prefix_length, host_routes=routes)
    mock_subnet_rep.update({'prefixlen': self.prefix_length, 'use_default_subnetpool': True})
    self._compare_subnets(mock_subnet_rep, subnet)
    self.assert_calls()