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
def send_flow_stats(self):
    """ Get all flow. """
    ofp = self.dp.ofproto
    parser = self.dp.ofproto_parser
    if ofp.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        req = parser.OFPFlowStatsRequest(self.dp, 0, parser.OFPMatch(), 255, ofp.OFPP_NONE)
    else:
        req = parser.OFPFlowStatsRequest(self.dp, 0, ofp.OFPTT_ALL, ofp.OFPP_ANY, ofp.OFPG_ANY, 0, 0, parser.OFPMatch())
    return self.send_msg(req)