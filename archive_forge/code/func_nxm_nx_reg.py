import sys
from struct import calcsize
from os_ken.lib import type_desc
from os_ken.ofproto.ofproto_common import OFP_HEADER_SIZE
from os_ken.ofproto import oxm_fields
def nxm_nx_reg(idx):
    return nxm_header(1, idx, 4)