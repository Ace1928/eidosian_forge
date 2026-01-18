import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@classmethod
def option_type(cls):
    return ND_OPTION_MTU