import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv6_dst(self, dp):
    field = dp.ofproto.OXM_OF_IPV6_DST
    ipv6_dst = '8893:65b3:6b49:3bdb:3d2:9401:866c:c96'
    value = self.ipv6_to_int(ipv6_dst)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(34525)
    self.add_set_field_action(dp, field, value, match)