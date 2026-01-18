import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_udp_dst(self, dp):
    field = dp.ofproto.OXM_OF_UDP_DST
    value = 17
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(ether.ETH_TYPE_IP)
    match.set_ip_proto(inet.IPPROTO_UDP)
    self.add_set_field_action(dp, field, value, match)