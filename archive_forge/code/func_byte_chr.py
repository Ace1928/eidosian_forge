import logging
import struct
def byte_chr(c):
    assert isinstance(c, int)
    return struct.pack('B', c)