from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_multiple_private_ip(self):
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [{'id': 'test-net-id', 'name': 'test-net'}]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': SUBNETS_WITH_NAT})])
    shared_mac = '11:22:33:44:55:66'
    distinct_mac = '66:55:44:33:22:11'
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'test-net': [{'OS-EXT-IPS:type': 'fixed', 'OS-EXT-IPS-MAC:mac_addr': distinct_mac, 'addr': '10.0.0.100', 'version': 4}, {'OS-EXT-IPS:type': 'fixed', 'OS-EXT-IPS-MAC:mac_addr': shared_mac, 'addr': '10.0.0.101', 'version': 4}], 'public': [{'OS-EXT-IPS:type': 'floating', 'OS-EXT-IPS-MAC:mac_addr': shared_mac, 'addr': PUBLIC_V4, 'version': 4}]})
    self.assertEqual('10.0.0.101', meta.get_server_private_ip(srv, self.cloud))
    self.assert_calls()