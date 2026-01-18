import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def recv_length(self):
    bits = self.header[frame_buffer._HEADER_LENGTH_INDEX]
    length_bits = bits & 127
    if length_bits == 126:
        v = self.recv_strict(2)
        self.length = struct.unpack('!H', v)[0]
    elif length_bits == 127:
        v = self.recv_strict(8)
        self.length = struct.unpack('!Q', v)[0]
    else:
        self.length = length_bits