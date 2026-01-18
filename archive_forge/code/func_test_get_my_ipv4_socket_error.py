import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch('socket.socket')
@mock.patch('oslo_utils.netutils._get_my_ipv4_address')
def test_get_my_ipv4_socket_error(self, ip, mock_socket):
    mock_socket.side_effect = socket.error
    ip.return_value = '1.2.3.4'
    addr = netutils.get_my_ipv4()
    self.assertEqual(addr, '1.2.3.4')