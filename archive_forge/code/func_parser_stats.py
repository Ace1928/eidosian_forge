import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
@classmethod
def parser_stats(cls, datapath, version, msg_type, msg_len, xid, buf):
    msg = MsgBase.parser.__func__(cls, datapath, version, msg_type, msg_len, xid, buf)
    msg.body = msg.parser_stats_body(msg.buf, msg.msg_len, ofproto.OFP_MULTIPART_REPLY_SIZE)
    return msg