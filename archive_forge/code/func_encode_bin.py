from .py3compat import int2byte
def encode_bin(data):
    """
    Create a binary representation of the given b'' object. Assume 8-bit
    ASCII. Example:

        >>> encode_bin('ab')
        b"\x00\x01\x01\x00\x00\x00\x00\x01\x00\x01\x01\x00\x00\x00\x01\x00"
    """
    return b''.join((_char_to_bin[ch] for ch in data))