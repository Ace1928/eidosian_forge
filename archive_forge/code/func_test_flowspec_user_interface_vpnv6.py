import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_flowspec_user_interface_vpnv6(self):
    rules = RULES_BASE + [bgp.FlowSpecIPv6DestPrefix(addr='2001::2', offset=32, length=128), bgp.FlowSpecIPv6SrcPrefix(addr='3002::3', length=128), bgp.FlowSpecNextHeader(operator=bgp.FlowSpecNextHeader.EQ, value=6), bgp.FlowSpecIPv6Fragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6FlowLabel(operator=bgp.FlowSpecIPv6FlowLabel.EQ, value=100)]
    msg = bgp.FlowSpecVPNv6NLRI.from_user(route_dist='65001:250', dst_prefix='2001::2/128/32', src_prefix='3002::3/128', next_header='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=LF', flow_label=100)
    msg2 = bgp.FlowSpecVPNv6NLRI(route_dist='65001:250', rules=rules)
    binmsg = msg.serialize()
    binmsg2 = msg2.serialize()
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
    msg3, rest = bgp.FlowSpecVPNv6NLRI.parser(binmsg)
    self.assertEqual(str(msg), str(msg3))
    self.assertEqual(rest, b'')