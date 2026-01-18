import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_mod_cookie_mask(self, dp):
    self._add_flow_for_flow_mod_tests(dp)
    action = dp.ofproto_parser.OFPActionOutput(3, 1500)
    self._verify[0][3] = action
    self._verify[1][3] = action
    cookie = 65535
    cookie_mask = 65280
    for table_id in range(2):
        self.mod_flow(dp, command=dp.ofproto.OFPFC_MODIFY, actions=[action], table_id=table_id, cookie=cookie, cookie_mask=cookie_mask)
    dp.send_barrier()
    self.send_flow_stats(dp)