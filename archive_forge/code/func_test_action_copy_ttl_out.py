import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_copy_ttl_out(self, dp):
    self._verify = [dp.ofproto.OFPAT_COPY_TTL_OUT]
    actions = [dp.ofproto_parser.OFPActionCopyTtlOut()]
    self.add_apply_actions(dp, actions)