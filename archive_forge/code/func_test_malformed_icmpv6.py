import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def test_malformed_icmpv6(self):
    m_short_buf = self.buf[1:self.icmp._MIN_LEN]
    self.assertRaises(struct.error, self.icmp.parser, m_short_buf)