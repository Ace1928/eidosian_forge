import sys
from struct import calcsize
from os_ken.lib import type_desc
from os_ken.ofproto.ofproto_common import OFP_HEADER_SIZE
from os_ken.ofproto import oxm_fields
def nxm_header__(vendor, field, hasmask, length):
    return vendor << 16 | field << 9 | hasmask << 8 | length