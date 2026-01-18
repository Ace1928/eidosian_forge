import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_nw_tos(self, dp):
    nw_tos = 1 << 2
    self._verify = [dp.ofproto.OFPAT_SET_NW_TOS, 'tos', nw_tos]
    action = dp.ofproto_parser.OFPActionSetNwTos(nw_tos)
    self.add_action(dp, [action])