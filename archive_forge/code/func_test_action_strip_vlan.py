import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_strip_vlan(self, dp):
    vlan_pcp = 4
    self._verify = [dp.ofproto.OFPAT_STRIP_VLAN, None, None]
    action = dp.ofproto_parser.OFPActionStripVlan()
    self.add_action(dp, [action])