from __future__ import annotations
from struct import pack, unpack_from
def o16be(i: int) -> bytes:
    return pack('>H', i)