import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@system_description.setter
def system_description(self, value):
    self.tlv_info = value