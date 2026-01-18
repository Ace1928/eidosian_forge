import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_vlan_vid(self, dp):
    vlan_vid = 2
    self._verify = [dp.ofproto.OFPAT_SET_VLAN_VID, 'vlan_vid', vlan_vid]
    action = dp.ofproto_parser.OFPActionVlanVid(vlan_vid)
    self.add_action(dp, [action])