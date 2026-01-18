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
def test_build_arp_vlan(self):
    p = self._build_arp(True)
    e = self.find_protocol(p, 'ethernet')
    self.assertTrue(e)
    self.assertEqual(e.ethertype, ether.ETH_TYPE_8021Q)
    v = self.find_protocol(p, 'vlan')
    self.assertTrue(v)
    self.assertEqual(v.ethertype, ether.ETH_TYPE_ARP)
    a = self.find_protocol(p, 'arp')
    self.assertTrue(a)
    self.assertEqual(a.hwtype, self.hwtype)
    self.assertEqual(a.proto, self.proto)
    self.assertEqual(a.hlen, self.hlen)
    self.assertEqual(a.plen, self.plen)
    self.assertEqual(a.opcode, self.opcode)
    self.assertEqual(a.src_mac, self.src_mac)
    self.assertEqual(a.src_ip, self.src_ip)
    self.assertEqual(a.dst_mac, self.dst_mac)
    self.assertEqual(a.dst_ip, self.dst_ip)