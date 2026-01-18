import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_generate_IPv6_with_error_prefix_type(self):
    mac = '00:16:3e:33:44:55'
    prefix = 123
    self.assertRaises(TypeError, lambda: netutils.get_ipv6_addr_by_EUI64(prefix, mac))