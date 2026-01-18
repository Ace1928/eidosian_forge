import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def ofp_role_from_user(self, role):
    return self._reserved_num_from_user(role, 'OFPCR_ROLE_')