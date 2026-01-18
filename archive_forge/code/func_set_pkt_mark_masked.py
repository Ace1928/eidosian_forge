import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def set_pkt_mark_masked(self, pkt_mark, mask):
    self.flow.pkt_mark = pkt_mark
    self.wc.pkt_mark_mask = mask