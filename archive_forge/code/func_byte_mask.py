import logging
import struct
def byte_mask(c, mask):
    assert isinstance(c, int)
    return struct.pack('B', c & mask)