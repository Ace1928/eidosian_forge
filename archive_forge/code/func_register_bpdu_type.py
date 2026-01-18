import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def register_bpdu_type(sub_cls):
    bpdu._BPDU_TYPES.setdefault(sub_cls.VERSION_ID, {})
    bpdu._BPDU_TYPES[sub_cls.VERSION_ID][sub_cls.BPDU_TYPE] = sub_cls
    return sub_cls