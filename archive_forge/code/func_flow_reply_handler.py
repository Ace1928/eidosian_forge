import sys
import logging
import itertools
from os_ken import utils
from os_ken.lib import mac
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller import handler
from os_ken.controller import dpset
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import CONFIG_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
@handler.set_ev_cls(ofp_event.EventOFPFlowStatsReply, handler.MAIN_DISPATCHER)
def flow_reply_handler(self, ev):
    self.run_verify(ev)