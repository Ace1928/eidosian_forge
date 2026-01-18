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
def set_ipv4_dst_masked(self, ipv4_dst, mask):
    self._wc.ft_set(ofproto.OFPXMT_OFB_IPV4_DST)
    self._flow.ipv4_dst = ipv4_dst
    self._wc.ipv4_dst_mask = mask