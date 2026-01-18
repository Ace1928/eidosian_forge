from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_external_provider_ipv4_neutron(self):
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks', json={'networks': [{'id': 'test-net-id', 'name': 'test-net', 'provider:network_type': 'vlan', 'provider:physical_network': 'vlan'}]}), dict(method='GET', uri='https://network.example.com/v2.0/subnets', json={'subnets': SUBNETS_WITH_NAT})])
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'test-net': [{'addr': PUBLIC_V4, 'version': 4}]})
    ip = meta.get_server_external_ipv4(cloud=self.cloud, server=srv)
    self.assertEqual(PUBLIC_V4, ip)
    self.assert_calls()