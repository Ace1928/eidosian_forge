import sys
from struct import calcsize
from os_ken.lib import type_desc
from os_ken.ofproto.ofproto_common import OFP_HEADER_SIZE
from os_ken.ofproto import oxm_fields
def nxm_nx_reg_w(idx):
    return nxm_header_w(1, idx, 4)