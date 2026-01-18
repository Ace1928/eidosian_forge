import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def nunpack(s: bytes, default: int=0) -> int:
    """Unpacks 1 to 4 or 8 byte integers (big endian)."""
    length = len(s)
    if not length:
        return default
    elif length == 1:
        return ord(s)
    elif length == 2:
        return cast(int, struct.unpack('>H', s)[0])
    elif length == 3:
        return cast(int, struct.unpack('>L', b'\x00' + s)[0])
    elif length == 4:
        return cast(int, struct.unpack('>L', s)[0])
    elif length == 8:
        return cast(int, struct.unpack('>Q', s)[0])
    else:
        raise TypeError('invalid length: %d' % length)