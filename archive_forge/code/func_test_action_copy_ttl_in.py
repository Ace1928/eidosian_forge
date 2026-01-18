import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_copy_ttl_in(self, dp):
    self._verify = [dp.ofproto.OFPAT_COPY_TTL_IN]
    actions = [dp.ofproto_parser.OFPActionCopyTtlIn()]
    self.add_apply_actions(dp, actions)