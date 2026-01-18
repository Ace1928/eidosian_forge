import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.arp import arp
from os_ken.lib.packet.vlan import vlan
from os_ken.lib import addrconv
def test_malformed_arp(self):
    m_short_buf = self.buf[1:arp._MIN_LEN]
    self.assertRaises(Exception, arp.parser, m_short_buf)