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
@set_ev_handler(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
def switch_features_handler(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    self.logger.debug('switch features ev %s', msg)
    datapath.id = msg.datapath_id
    if datapath.ofproto.OFP_VERSION < 4:
        datapath.ports = msg.ports
    else:
        datapath.ports = {}
    if datapath.ofproto.OFP_VERSION < 4:
        self.logger.debug('move onto main mode')
        ev.msg.datapath.set_state(MAIN_DISPATCHER)
    else:
        port_desc = datapath.ofproto_parser.OFPPortDescStatsRequest(datapath, 0)
        datapath.send_msg(port_desc)