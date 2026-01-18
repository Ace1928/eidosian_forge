from __future__ import annotations
from struct import pack, unpack_from
def o8(i: int) -> bytes:
    return bytes((i & 255,))