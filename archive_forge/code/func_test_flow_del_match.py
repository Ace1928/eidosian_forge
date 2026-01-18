import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_del_match(self, dp):
    self._add_flow_for_flow_mod_tests(dp)
    del self._verify[3]
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_dst(b'\xff' * 6)
    self.mod_flow(dp, command=dp.ofproto.OFPFC_DELETE, table_id=dp.ofproto.OFPTT_ALL, match=match)
    dp.send_barrier()
    self.send_flow_stats(dp)