import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def encode_variable_int(value):
    """Encode variable length integer.

    Returns the integer as a list of bytes,
    where the last byte is < 128.

    This is used for delta times and meta message payload
    length.
    """
    if not isinstance(value, Integral) or value < 0:
        raise ValueError('variable int must be a non-negative integer')
    bytes = []
    while value:
        bytes.append(value & 127)
        value >>= 7
    if bytes:
        bytes.reverse()
        for i in range(len(bytes) - 1):
            bytes[i] |= 128
        return bytes
    else:
        return [0]