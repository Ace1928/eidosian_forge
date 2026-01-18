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
def test_flowspec_user_interface_l2vpn(self):
    rules = RULES_L2VPN_BASE
    msg = bgp.FlowSpecL2VPNNLRI.from_user(route_dist='65001:250', ether_type=2048, src_mac='12:34:56:78:90:AB', dst_mac='BE:EF:C0:FF:EE:DD', llc_dsap=66, llc_ssap=66, llc_control=100, snap=74565, vlan_id='>4000', vlan_cos='>=3', inner_vlan_id='<3000', inner_vlan_cos='<=5')
    msg2 = bgp.FlowSpecL2VPNNLRI(route_dist='65001:250', rules=rules)
    binmsg = msg.serialize()
    binmsg2 = msg2.serialize()
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
    msg3, rest = bgp.FlowSpecL2VPNNLRI.parser(binmsg)
    self.assertEqual(str(msg), str(msg3))
    self.assertEqual(rest, b'')