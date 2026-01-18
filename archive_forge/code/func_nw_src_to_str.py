import struct
import socket
import logging
from os_ken.ofproto import ofproto_v1_0
from os_ken.lib import ofctl_utils
from os_ken.lib.mac import haddr_to_bin, haddr_to_str
def nw_src_to_str(wildcards, addr):
    ip = socket.inet_ntoa(struct.pack('!I', addr))
    mask = 32 - ((wildcards & ofproto_v1_0.OFPFW_NW_SRC_MASK) >> ofproto_v1_0.OFPFW_NW_SRC_SHIFT)
    if mask == 32:
        mask = 0
    if mask:
        ip += '/%d' % mask
    return ip