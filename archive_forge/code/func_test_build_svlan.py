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
def test_build_svlan(self):
    p = self._build_svlan()
    e = self.find_protocol(p, 'ethernet')
    self.assertTrue(e)
    self.assertEqual(e.ethertype, ether.ETH_TYPE_8021AD)
    sv = self.find_protocol(p, 'svlan')
    self.assertTrue(sv)
    self.assertEqual(sv.ethertype, ether.ETH_TYPE_8021Q)
    v = self.find_protocol(p, 'vlan')
    self.assertTrue(v)
    self.assertEqual(v.ethertype, ether.ETH_TYPE_IP)
    ip = self.find_protocol(p, 'ipv4')
    self.assertTrue(ip)
    self.assertEqual(sv.pcp, self.pcp)
    self.assertEqual(sv.cfi, self.cfi)
    self.assertEqual(sv.vid, self.vid)
    self.assertEqual(sv.ethertype, self.ethertype)