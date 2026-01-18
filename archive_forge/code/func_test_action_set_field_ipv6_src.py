import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv6_src(self, dp):
    field = dp.ofproto.OXM_OF_IPV6_SRC
    ipv6_src = '7527:c798:c772:4a18:117a:14ff:c1b6:e4ef'
    value = self.ipv6_to_int(ipv6_src)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(34525)
    self.add_set_field_action(dp, field, value, match)