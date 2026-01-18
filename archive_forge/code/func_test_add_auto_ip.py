from unittest.mock import patch
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@patch.object(connection.Connection, 'get_floating_ip')
@patch.object(connection.Connection, '_attach_ip_to_server')
@patch.object(connection.Connection, 'available_floating_ip')
def test_add_auto_ip(self, mock_available_floating_ip, mock_attach_ip_to_server, mock_get_floating_ip):
    server_dict = fakes.make_fake_server(server_id='server-id', name='test-server', status='ACTIVE', addresses={})
    floating_ip_dict = {'id': 'this-is-a-floating-ip-id', 'fixed_ip_address': None, 'internal_network': None, 'floating_ip_address': '203.0.113.29', 'network': 'this-is-a-net-or-pool-id', 'attached': False, 'status': 'ACTIVE'}
    mock_available_floating_ip.return_value = floating_ip_dict
    self.cloud.add_auto_ip(server=server_dict)
    mock_attach_ip_to_server.assert_called_with(timeout=60, wait=False, server=server_dict, floating_ip=floating_ip_dict, skip_attach=False)