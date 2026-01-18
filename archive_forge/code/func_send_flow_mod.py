import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def send_flow_mod(self, dp, rule, cookie, command, idle_timeout, hard_timeout, priority=None, buffer_id=4294967295, out_port=None, flags=0, actions=None):
    if priority is None:
        priority = dp.ofproto.OFP_DEFAULT_PRIORITY
    if out_port is None:
        out_port = dp.ofproto.OFPP_NONE
    match_tuple = rule.match_tuple()
    match = dp.ofproto_parser.OFPMatch(*match_tuple)
    m = dp.ofproto_parser.OFPFlowMod(dp, match, cookie, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, flags, actions)
    dp.send_msg(m)