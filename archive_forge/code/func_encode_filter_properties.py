import bz2
import lzma
import struct
import sys
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import bcj
import inflate64
import pyppmd
import pyzstd
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.helpers import Buffer, calculate_crc32, calculate_key
from py7zr.properties import (
@classmethod
def encode_filter_properties(cls, filter: Dict[str, Union[str, int]]):
    order = filter.get('order', 8)
    mem = filter.get('mem', 24)
    if isinstance(mem, str):
        if mem.isdecimal():
            size = 1 << int(mem)
        elif mem.lower().endswith('m') and mem[:-1].isdecimal():
            size = int(mem[:-1]) << 20
        elif mem.lower().endswith('k') and mem[:-1].isdecimal():
            size = int(mem[:-1]) << 10
        elif mem.lower().endswith('b') and mem[:-1].isdecimal():
            size = int(mem[:-1])
        else:
            raise ValueError('Ppmd:Unsupported memory size is specified: {0}'.format(mem))
    elif isinstance(mem, int):
        size = 1 << mem
    else:
        raise ValueError('Ppmd:Unsupported memory size is specified: {0}'.format(mem))
    properties = struct.pack('<BLBB', order, size, 0, 0)
    return properties