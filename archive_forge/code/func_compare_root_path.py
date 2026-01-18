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
@staticmethod
def compare_root_path(path_cost1, path_cost2, bridge_id1, bridge_id2, port_id1, port_id2):
    """ Decide the port of the side near a root bridge.
            It is compared by the following priorities.
             1. root path cost
             2. designated bridge ID value
             3. designated port ID value """
    result = Stp._cmp_value(path_cost1, path_cost2)
    if not result:
        result = Stp._cmp_value(bridge_id1, bridge_id2)
        if not result:
            result = Stp._cmp_value(port_id1, port_id2)
    return result