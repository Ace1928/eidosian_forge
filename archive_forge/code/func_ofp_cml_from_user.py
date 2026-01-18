import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def ofp_cml_from_user(self, max_len):
    return self._reserved_num_from_user(max_len, 'OFPCML_')