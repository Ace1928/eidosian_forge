import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
def test_build_igmp(self):
    p = self._build_igmp()
    e = self.find_protocol(p, 'ethernet')
    self.assertTrue(e)
    self.assertEqual(e.ethertype, ether.ETH_TYPE_IP)
    i = self.find_protocol(p, 'ipv4')
    self.assertTrue(i)
    self.assertEqual(i.proto, inet.IPPROTO_IGMP)
    g = self.find_protocol(p, 'igmpv3_report')
    self.assertTrue(g)
    self.assertEqual(g.msgtype, self.msgtype)
    self.assertEqual(g.csum, checksum(self.buf))
    self.assertEqual(g.record_num, self.record_num)
    self.assertEqual(g.records, self.records)