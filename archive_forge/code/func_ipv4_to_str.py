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
def ipv4_to_str(self, integre):
    ip_list = [str(integre >> 24 - n * 8 & 255) for n in range(4)]
    return '.'.join(ip_list)