from .. import osutils
def decode_base128_int(data):
    """Decode an integer from a 7-bit lsb encoding."""
    offset = 0
    val = 0
    shift = 0
    bval = data[offset]
    while bval >= 128:
        val |= (bval & 127) << shift
        shift += 7
        offset += 1
        bval = data[offset]
    val |= bval << shift
    offset += 1
    return (val, offset)