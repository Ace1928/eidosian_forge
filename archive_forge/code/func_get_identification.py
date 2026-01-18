import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
def get_identification(self):
    self.identification += 1
    self.identification &= 65535
    if self.identification == 0:
        self.identification += 1
        self.identification &= 65535
    return self.identification