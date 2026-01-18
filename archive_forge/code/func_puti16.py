from __future__ import annotations
import os
from typing import BinaryIO
from . import Image, _binary
def puti16(fp: BinaryIO, values: tuple[int, int, int, int, int, int, int, int, int, int]) -> None:
    """Write network order (big-endian) 16-bit sequence"""
    for v in values:
        if v < 0:
            v += 65536
        fp.write(_binary.o16be(v))