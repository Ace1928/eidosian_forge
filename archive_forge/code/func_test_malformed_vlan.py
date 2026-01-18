import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.vlan import vlan
from os_ken.lib.packet.vlan import svlan
def test_malformed_vlan(self):
    m_short_buf = self.buf[1:vlan._MIN_LEN]
    self.assertRaises(Exception, vlan.parser, m_short_buf)