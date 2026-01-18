import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv6_nd_tll(self, dp):
    field = dp.ofproto.OXM_OF_IPV6_ND_TLL
    tll = '83:13:48:1e:d0:b0'
    value = self.haddr_to_bin(tll)
    self.add_set_field_action(dp, field, value)