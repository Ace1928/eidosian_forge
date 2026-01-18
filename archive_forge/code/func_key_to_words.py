import struct
from passlib.utils import repeat_string
@staticmethod
def key_to_words(data, size=18):
    """convert data to tuple of <size> 4-byte integers, repeating or
        truncating data as needed to reach specified size"""
    assert isinstance(data, bytes)
    dlen = len(data)
    if not dlen:
        return [0] * size
    data = repeat_string(data, size << 2)
    return struct.unpack('>%dI' % (size,), data)