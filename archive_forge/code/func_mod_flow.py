import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def mod_flow(self, dp, cookie=0, cookie_mask=0, table_id=0, command=None, idle_timeout=0, hard_timeout=0, priority=255, buffer_id=4294967295, match=None, actions=None, inst_type=None, out_port=None, out_group=None, flags=0, inst=None):
    if command is None:
        command = dp.ofproto.OFPFC_ADD
    if inst is None:
        if inst_type is None:
            inst_type = dp.ofproto.OFPIT_APPLY_ACTIONS
        inst = []
        if actions is not None:
            inst = [dp.ofproto_parser.OFPInstructionActions(inst_type, actions)]
    if match is None:
        match = dp.ofproto_parser.OFPMatch()
    if out_port is None:
        out_port = dp.ofproto.OFPP_ANY
    if out_group is None:
        out_group = dp.ofproto.OFPG_ANY
    m = dp.ofproto_parser.OFPFlowMod(dp, cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, match, inst)
    dp.send_msg(m)