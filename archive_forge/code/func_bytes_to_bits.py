import re as _re
def bytes_to_bits():
    """
    :return: A 256 element list containing 8-bit binary digit strings. The
        list index value is equivalent to its bit string value.
    """
    lookup = []
    bits_per_byte = range(7, -1, -1)
    for num in range(256):
        bits = 8 * [None]
        for i in bits_per_byte:
            bits[i] = '01'[num & 1]
            num >>= 1
        lookup.append(''.join(bits))
    return lookup