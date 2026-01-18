from __future__ import annotations
from struct import pack, unpack_from
def i16be(c: bytes, o: int=0) -> int:
    return unpack_from('>H', c, o)[0]