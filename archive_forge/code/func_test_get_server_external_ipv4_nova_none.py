from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_server_external_ipv4_nova_none(self):
    self.cloud.config.config['has_network'] = False
    srv = fakes.make_fake_server(server_id='test-id', name='test-name', status='ACTIVE', addresses={'test-net': [{'addr': PRIVATE_V4}]})
    ip = meta.get_server_external_ipv4(cloud=self.cloud, server=srv)
    self.assertIsNone(ip)