import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_mpls_label(self, dp):
    field = dp.ofproto.OXM_OF_MPLS_LABEL
    value = 76
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(ether.ETH_TYPE_MPLS)
    self.add_set_field_action(dp, field, value, match)