import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch('socket.socket')
@mock.patch('oslo_utils.netutils._get_my_ipv6_address')
def test_get_my_ipv6_socket_error(self, ip, mock_socket):
    mock_socket.side_effect = socket.error
    ip.return_value = '2001:db8::2'
    addr = netutils.get_my_ipv6()
    self.assertEqual(addr, '2001:db8::2')