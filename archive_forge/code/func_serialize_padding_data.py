import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
def serialize_padding_data(self):
    if 'PADDED' in self.flags:
        return _STRUCT_B.pack(self.pad_length)
    return b''