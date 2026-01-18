from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def serialise_header_fields(d, endianness):
    l = [(i.value, (header_field_codes[i], v)) for i, v in sorted(d.items())]
    return _header_fields_type.serialise(l, 12, endianness)