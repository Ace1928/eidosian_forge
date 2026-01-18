import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_nw_src(self, dp):
    nw_src = '216.132.81.105'
    nw_src_int = self.ipv4_to_int(nw_src)
    self._verify = [dp.ofproto.OFPAT_SET_NW_SRC, 'nw_addr', nw_src_int]
    action = dp.ofproto_parser.OFPActionSetNwSrc(nw_src_int)
    self.add_action(dp, [action])