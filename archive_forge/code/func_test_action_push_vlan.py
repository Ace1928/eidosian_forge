import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_push_vlan(self, dp):
    ethertype = ether.ETH_TYPE_8021Q
    self._verify = [dp.ofproto.OFPAT_PUSH_VLAN, 'ethertype', ethertype]
    actions = [dp.ofproto_parser.OFPActionPushVlan(ethertype)]
    self.add_apply_actions(dp, actions)