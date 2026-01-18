import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_pop_vlan(self, dp):
    self._verify = [dp.ofproto.OFPAT_POP_VLAN]
    actions = [dp.ofproto_parser.OFPActionPopVlan()]
    match = dp.ofproto_parser.OFPMatch()
    match.set_vlan_vid(1)
    self.add_apply_actions(dp, actions, match)