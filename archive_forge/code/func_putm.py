import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def putm(self, buf, offset, value, mask):
    if mask == 0:
        return 0
    elif self._is_all_ones(mask):
        return self._put(buf, offset, value)
    else:
        return self.putw(buf, offset, value, mask)