import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_aggregate_stats_request(self, dp):
    self._verify = dp.ofproto.OFPST_AGGREGATE
    match = dp.ofproto_parser.OFPMatch()
    m = dp.ofproto_parser.OFPAggregateStatsRequest(dp, dp.ofproto.OFPTT_ALL, dp.ofproto.OFPP_ANY, dp.ofproto.OFPG_ANY, 0, 0, match)
    dp.send_msg(m)