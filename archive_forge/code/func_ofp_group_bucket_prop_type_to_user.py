import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def ofp_group_bucket_prop_type_to_user(self, group):
    return self._reserved_num_to_user(group, 'OFPGBPT_')