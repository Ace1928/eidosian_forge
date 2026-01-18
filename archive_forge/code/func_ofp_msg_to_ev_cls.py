import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event
def ofp_msg_to_ev_cls(msg_cls):
    name = _ofp_msg_name_to_ev_name(msg_cls.__name__)
    return _OFP_MSG_EVENTS[name]