import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def set_dl_vlan_pcp(self, dl_vlan_pcp):
    self.wc.wildcards &= ~ofproto_v1_0.OFPFW_DL_VLAN_PCP
    self.flow.dl_vlan_pcp = dl_vlan_pcp