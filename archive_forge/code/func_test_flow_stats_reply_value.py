import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_stats_reply_value(self, dp):
    self._verify = []
    c = 0
    while c < self.n_tables:
        v = (c, c + 1, c + 2, c + 3, c + 4)
        self._verify.append(v)
        self.mod_flow(dp, table_id=v[0], cookie=v[1], idle_timeout=v[2], hard_timeout=v[3], priority=v[4])
        c += 1
    dp.send_barrier()
    self.send_flow_stats(dp)