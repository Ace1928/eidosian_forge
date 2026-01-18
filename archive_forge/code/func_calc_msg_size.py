from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def calc_msg_size(buf):
    endian, = struct.unpack('c', buf[:1])
    endian = endian_map[endian]
    body_length, = struct.unpack(endian.struct_code() + 'I', buf[4:8])
    fields_array_len, = struct.unpack(endian.struct_code() + 'I', buf[12:16])
    header_len = 16 + fields_array_len
    return header_len + padding(header_len, 8) + body_length