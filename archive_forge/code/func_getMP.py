import struct
from cryptography.utils import int_to_bytes
from twisted.python.deprecate import deprecated
from twisted.python.versions import Version
def getMP(data, count=1):
    """
    Get multiple precision integer out of the string.  A multiple precision
    integer is stored as a 4-byte length followed by length bytes of the
    integer.  If count is specified, get count integers out of the string.
    The return value is a tuple of count integers followed by the rest of
    the data.
    """
    mp = []
    c = 0
    for i in range(count):
        length, = struct.unpack('>L', data[c:c + 4])
        mp.append(int.from_bytes(data[c + 4:c + 4 + length], 'big'))
        c += 4 + length
    return tuple(mp) + (data[c:],)