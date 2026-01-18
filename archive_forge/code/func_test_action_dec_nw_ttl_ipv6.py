import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_dec_nw_ttl_ipv6(self, dp):
    self._verify = [dp.ofproto.OFPAT_DEC_NW_TTL]
    actions = [dp.ofproto_parser.OFPActionDecNwTtl()]
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(34525)
    self.add_apply_actions(dp, actions, match)