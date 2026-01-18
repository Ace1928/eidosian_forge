from .py3compat import int2byte
def int_to_bin(number, width=32):
    """
    Convert an integer into its binary representation in a bytes object.
    Width is the amount of bits to generate. If width is larger than the actual
    amount of bits required to represent number in binary, sign-extension is
    used. If it's smaller, the representation is trimmed to width bits.
    Each "bit" is either '\\x00' or '\\x01'. The MSBit is first.

    Examples:

        >>> int_to_bin(19, 5)
        b'\\x01\\x00\\x00\\x01\\x01'
        >>> int_to_bin(19, 8)
        b'\\x00\\x00\\x00\\x01\\x00\\x00\\x01\\x01'
    """
    if number < 0:
        number += 1 << width
    i = width - 1
    bits = bytearray(width)
    while number and i >= 0:
        bits[i] = number & 1
        number >>= 1
        i -= 1
    return bytes(bits)