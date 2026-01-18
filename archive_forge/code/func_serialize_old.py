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
def serialize_old(self, buf, offset):
    len_ = ofproto.OFP_ACTION_SET_FIELD_SIZE + self.field.oxm_len()
    self.len = utils.round_up(len_, 8)
    pad_len = self.len - len_
    msg_pack_into('!HH', buf, offset, self.type, self.len)
    self.field.serialize(buf, offset + 4)
    offset += len_
    msg_pack_into('%dx' % pad_len, buf, offset)