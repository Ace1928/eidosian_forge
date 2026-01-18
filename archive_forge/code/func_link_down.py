import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def link_down(self, ofp_port):
    """ DESIGNATED_PORT/NON_DESIGNATED_PORT: change status to DISABLE.
            ROOT_PORT: change status to DISABLE and recalculate STP. """
    port = self.ports[ofp_port.port_no]
    init_stp_flg = bool(port.role is ROOT_PORT)
    port.down(PORT_STATE_DISABLE, msg_init=True)
    self.ports_state[ofp_port.port_no] = ofp_port.state
    if init_stp_flg:
        self.recalculate_spanning_tree()