import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
def ofp_msg_to_ev(msg):
    return ofp_msg_to_ev_cls(msg.__class__)(msg)