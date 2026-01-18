import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import inet
from os_ken.lib.packet import tcp
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def test_malformed_tcp(self):
    m_short_buf = self.buf[1:tcp.tcp._MIN_LEN]
    self.assertRaises(Exception, tcp.tcp.parser, m_short_buf)