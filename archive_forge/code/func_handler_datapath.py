import logging
import struct
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller import dpset
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.mac import haddr_to_str
@set_ev_cls(dpset.EventDP, dpset.DPSET_EV_DISPATCHER)
def handler_datapath(self, ev):
    if ev.enter:
        self._define_flow(ev.dp)