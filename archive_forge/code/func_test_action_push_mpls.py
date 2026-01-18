import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_push_mpls(self, dp):
    ethertype = ether.ETH_TYPE_MPLS
    self._verify = [dp.ofproto.OFPAT_PUSH_MPLS, 'ethertype', ethertype]
    actions = [dp.ofproto_parser.OFPActionPushMpls(ethertype)]
    self.add_apply_actions(dp, actions)