import re
import struct
def ascii85decode(data: bytes) -> bytes:
    """
    In ASCII85 encoding, every four bytes are encoded with five ASCII
    letters, using 85 different types of characters (as 256**4 < 85**5).
    When the length of the original bytes is not a multiple of 4, a special
    rule is used for round up.

    The Adobe's ASCII85 implementation is slightly different from
    its original in handling the last characters.

    """
    n = b = 0
    out = b''
    for i in iter(data):
        c = bytes((i,))
        if b'!' <= c and c <= b'u':
            n += 1
            b = b * 85 + (ord(c) - 33)
            if n == 5:
                out += struct.pack('>L', b)
                n = b = 0
        elif c == b'z':
            assert n == 0, str(n)
            out += b'\x00\x00\x00\x00'
        elif c == b'~':
            if n:
                for _ in range(5 - n):
                    b = b * 85 + 84
                out += struct.pack('>L', b)[:n - 1]
            break
    return out