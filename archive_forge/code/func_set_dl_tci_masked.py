import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def set_dl_tci_masked(self, tci, mask):
    self.wc.vlan_tci_mask = mask
    self.flow.vlan_tci = tci