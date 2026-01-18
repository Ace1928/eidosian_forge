import itertools
import logging
import warnings
import os_ken.base.app_manager
from os_ken.lib import hub
from os_ken import utils
from os_ken.controller import ofp_event
from os_ken.controller.controller import OpenFlowController
from os_ken.controller.handler import set_ev_handler
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER,\
from os_ken.ofproto import ofproto_parser
@set_ev_handler(ofp_event.EventOFPEchoRequest, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
def echo_request_handler(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    echo_reply = datapath.ofproto_parser.OFPEchoReply(datapath)
    echo_reply.xid = msg.xid
    echo_reply.data = msg.data
    datapath.send_msg(echo_reply)