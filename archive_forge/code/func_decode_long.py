from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def decode_long(data):
    """Decode a long from a two's complement little-endian binary string.

    >>> decode_long(b'')
    0
    >>> decode_long(b"\\xff\\x00")
    255
    >>> decode_long(b"\\xff\\x7f")
    32767
    >>> decode_long(b"\\x00\\xff")
    -256
    >>> decode_long(b"\\x00\\x80")
    -32768
    >>> decode_long(b"\\x80")
    -128
    >>> decode_long(b"\\x7f")
    127
    """
    return int.from_bytes(data, byteorder='little', signed=True)