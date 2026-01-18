from __future__ import annotations
import io
from typing import BinaryIO, Callable
from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32
def sz(s: bytes, o: int) -> bytes:
    return s[o:s.index(b'\x00', o)]