import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def ofp_buffer_from_user(self, buffer):
    if buffer in ['OFP_NO_BUFFER', 'NO_BUFFER']:
        return self.ofproto.OFP_NO_BUFFER
    else:
        return buffer