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
@set_ev_cls(ofp_event.EventOFPBarrierReply, handler.MAIN_DISPATCHER)
def barrier_reply_handler(self, ev):
    state_list = [STATE_INIT_FLOW, STATE_INIT_THROUGHPUT_FLOW, STATE_INIT_METER, STATE_INIT_GROUP, STATE_FLOW_INSTALL, STATE_THROUGHPUT_FLOW_INSTALL, STATE_METER_INSTALL, STATE_GROUP_INSTALL, STATE_SEND_BARRIER]
    if self.state in state_list:
        if self.waiter and ev.msg.xid in self.send_msg_xids:
            self.rcv_msgs.append(ev.msg)
            self.waiter.set()
            hub.sleep(0)