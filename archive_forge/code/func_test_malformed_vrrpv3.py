import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
def test_malformed_vrrpv3(self):
    m_short_buf = self.buf[1:vrrp.vrrpv3._MIN_LEN]
    self.assertRaises(Exception, vrrp.vrrp.parser, m_short_buf)