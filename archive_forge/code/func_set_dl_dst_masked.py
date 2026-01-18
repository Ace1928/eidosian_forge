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
def set_dl_dst_masked(self, dl_dst, mask):
    self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_DST)
    self._wc.dl_dst_mask = mask
    self._flow.dl_dst = mac.haddr_bitand(dl_dst, mask)