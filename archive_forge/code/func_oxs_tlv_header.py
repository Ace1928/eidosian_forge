from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_utils
from os_ken.ofproto import oxm_fields
from os_ken.ofproto import oxs_fields
from struct import calcsize
def oxs_tlv_header(field, length):
    return _oxs_tlv_header(OFPXSC_OPENFLOW_BASIC, field, 0, length)