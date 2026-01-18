from __future__ import annotations
from struct import pack, unpack_from
def i32le(c: bytes, o: int=0) -> int:
    """
    Converts a 4-bytes (32 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from('<I', c, o)[0]