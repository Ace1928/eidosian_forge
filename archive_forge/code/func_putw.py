import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def putw(self, buf, offset, value, mask):
    len_ = self._put(buf, offset, value)
    return len_ + self._put(buf, offset + len_, mask)