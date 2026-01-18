import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_removed_delete(self, dp):
    reason = dp.ofproto.OFPRR_DELETE
    self._add_flow_flow_removed(dp, reason)
    dp.send_barrier()
    self.delete_all_flows(dp)