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
@set_ev_handler(ofp_event.EventOFPPortDescStatsReply, CONFIG_DISPATCHER)
def multipart_reply_handler(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for port in msg.body:
            datapath.ports[port.port_no] = port
    if msg.flags & datapath.ofproto.OFPMPF_REPLY_MORE:
        return
    self.logger.debug('move onto main mode')
    ev.msg.datapath.set_state(MAIN_DISPATCHER)