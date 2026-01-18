import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@system_name.setter
def system_name(self, value):
    self.tlv_info = value