import inspect
import logging
import struct
import unittest
from os_ken.lib import addrconv
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import sctp
from os_ken.ofproto import ether
from os_ken.ofproto import inet
def test_build_sctp(self):
    eth = ethernet.ethernet('00:aa:aa:aa:aa:aa', '00:bb:bb:bb:bb:bb', ether.ETH_TYPE_IP)
    ip4 = ipv4.ipv4(4, 5, 16, 0, 0, 2, 0, 64, inet.IPPROTO_SCTP, 0, '192.168.1.1', '10.144.1.1')
    pkt = eth / ip4 / self.sc
    eth = pkt.get_protocol(ethernet.ethernet)
    self.assertTrue(eth)
    self.assertEqual(eth.ethertype, ether.ETH_TYPE_IP)
    ip4 = pkt.get_protocol(ipv4.ipv4)
    self.assertTrue(ip4)
    self.assertEqual(ip4.proto, inet.IPPROTO_SCTP)
    sc = pkt.get_protocol(sctp.sctp)
    self.assertTrue(sc)
    self.assertEqual(sc, self.sc)