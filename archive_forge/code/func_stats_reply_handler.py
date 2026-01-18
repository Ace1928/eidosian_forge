import binascii
import inspect
import json
import logging
import math
import netaddr
import os
import signal
import sys
import time
import traceback
from random import randint
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import hub
from os_ken.lib import stringify
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
@set_ev_cls([ofp_event.EventOFPFlowStatsReply, ofp_event.EventOFPMeterConfigStatsReply, ofp_event.EventOFPTableStatsReply, ofp_event.EventOFPPortStatsReply, ofp_event.EventOFPGroupDescStatsReply], handler.MAIN_DISPATCHER)
def stats_reply_handler(self, ev):
    ofp = ev.msg.datapath.ofproto
    event_states = {ofp_event.EventOFPFlowStatsReply: [STATE_FLOW_EXIST_CHK, STATE_THROUGHPUT_FLOW_EXIST_CHK, STATE_GET_THROUGHPUT], ofp_event.EventOFPTableStatsReply: [STATE_GET_MATCH_COUNT, STATE_FLOW_UNMATCH_CHK], ofp_event.EventOFPPortStatsReply: [STATE_TARGET_PKT_COUNT, STATE_TESTER_PKT_COUNT]}
    if ofp.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
        event_states[ofp_event.EventOFPGroupDescStatsReply] = [STATE_GROUP_EXIST_CHK]
    if ofp.OFP_VERSION >= ofproto_v1_3.OFP_VERSION:
        event_states[ofp_event.EventOFPMeterConfigStatsReply] = [STATE_METER_EXIST_CHK]
    if self.state in event_states[ev.__class__]:
        if self.waiter and ev.msg.xid in self.send_msg_xids:
            self.rcv_msgs.append(ev.msg)
            if not ev.msg.flags:
                self.waiter.set()
                hub.sleep(0)