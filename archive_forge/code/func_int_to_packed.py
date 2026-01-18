import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def int_to_packed(int_val):
    """
    :param int_val: the integer to be packed.

    :return: a packed string that is equivalent to value represented by an
    unsigned integer.
    """
    words = int_to_words(int_val)
    return _struct.pack('>8B', *words)