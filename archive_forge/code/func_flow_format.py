import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def flow_format(self):
    if self.wc.tun_id_mask != 0:
        return ofproto_v1_0.NXFF_NXM
    if self.wc.dl_dst_mask:
        return ofproto_v1_0.NXFF_NXM
    if self.wc.dl_src_mask:
        return ofproto_v1_0.NXFF_NXM
    if not self.wc.wildcards & FWW_NW_ECN:
        return ofproto_v1_0.NXFF_NXM
    if self.wc.regs_bits > 0:
        return ofproto_v1_0.NXFF_NXM
    if self.flow.tcp_flags > 0:
        return ofproto_v1_0.NXFF_NXM
    return ofproto_v1_0.NXFF_OPENFLOW10