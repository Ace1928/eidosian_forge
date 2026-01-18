import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def test_generate_IPv6_by_EUI64(self):
    addr = netutils.get_ipv6_addr_by_EUI64('2001:db8::', '00:16:3e:33:44:55')
    self.assertEqual('2001:db8::216:3eff:fe33:4455', addr.format())