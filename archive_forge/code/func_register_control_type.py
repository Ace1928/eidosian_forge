import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
@staticmethod
def register_control_type(register_cls):
    llc._CTR_TYPES[register_cls.TYPE] = register_cls
    return register_cls