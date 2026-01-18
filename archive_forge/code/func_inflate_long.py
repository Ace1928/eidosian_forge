import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def inflate_long(s, always_positive=False):
    """turns a normalized byte string into a long-int
    (adapted from Crypto.Util.number)"""
    out = 0
    negative = 0
    if not always_positive and len(s) > 0 and (byte_ord(s[0]) >= 128):
        negative = 1
    if len(s) % 4:
        filler = zero_byte
        if negative:
            filler = max_byte
        s = filler * (4 - len(s) % 4) + s
    for i in range(0, len(s), 4):
        out = (out << 32) + struct.unpack('>I', s[i:i + 4])[0]
    if negative:
        out -= 1 << 8 * len(s)
    return out