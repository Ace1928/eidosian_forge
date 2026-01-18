from .. import osutils
def encode_base128_int(val):
    """Convert an integer into a 7-bit lsb encoding."""
    data = bytearray()
    count = 0
    while val >= 128:
        data.append((val | 128) & 255)
        val >>= 7
    data.append(val)
    return bytes(data)