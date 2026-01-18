import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
def parse_padding_data(self, data):
    if 'PADDED' in self.flags:
        try:
            self.pad_length = struct.unpack('!B', data[:1])[0]
        except struct.error:
            raise InvalidFrameError('Invalid Padding data')
        return 1
    return 0