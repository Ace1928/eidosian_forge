import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def ofp_table_to_user(self, table):
    return self._reserved_num_to_user(table, 'OFPTT_')