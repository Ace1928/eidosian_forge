import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
@staticmethod
def register_queue_property(prop_type, prop_len):

    def _register_queue_propery(cls):
        cls.cls_prop_type = prop_type
        cls.cls_prop_len = prop_len
        OFPQueuePropHeader._QUEUE_PROPERTIES[prop_type] = cls
        return cls
    return _register_queue_propery