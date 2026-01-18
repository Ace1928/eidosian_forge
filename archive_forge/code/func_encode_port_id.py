import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def encode_port_id(priority, port_number):
    return (priority << 8) + port_number