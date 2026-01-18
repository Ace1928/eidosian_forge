import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@classmethod
def chunk_type(cls):
    return TYPE_SHUTDOWN_COMPLETE