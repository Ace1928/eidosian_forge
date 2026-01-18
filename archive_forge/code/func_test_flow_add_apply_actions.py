import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_add_apply_actions(self, dp):
    inst_type = dp.ofproto.OFPIT_APPLY_ACTIONS
    self._verify = inst_type
    actions = [dp.ofproto_parser.OFPActionOutput(1, 1500)]
    self.mod_flow(dp, actions=actions, inst_type=inst_type)
    self.send_flow_stats(dp)