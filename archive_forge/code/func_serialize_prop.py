import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@classmethod
def serialize_prop(cls, pause):
    data = bytearray()
    msg_pack_into('!HH4x', data, 0, nicira_ext.NXAC2PT_PAUSE, 4)
    return data