import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def mf_from_nxm_header(nxm_header):
    if nxm_header not in _MF_FIELDS:
        return None
    make = _MF_FIELDS.get(nxm_header)
    assert make is not None
    return make(nxm_header)