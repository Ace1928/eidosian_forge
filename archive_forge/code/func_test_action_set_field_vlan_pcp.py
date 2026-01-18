import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_vlan_pcp(self, dp):
    field = dp.ofproto.OXM_OF_VLAN_PCP
    value = 3
    match = dp.ofproto_parser.OFPMatch()
    match.set_vlan_vid(1)
    self.add_set_field_action(dp, field, value, match)