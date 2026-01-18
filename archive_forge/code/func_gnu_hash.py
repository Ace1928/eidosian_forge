import struct
from ..common.utils import struct_parse
from .sections import Section
@staticmethod
def gnu_hash(key):
    """ Compute the GNU-style hash value for a given symbol name.
        """
    if not isinstance(key, bytes):
        key = key.encode('utf-8')
    h = 5381
    for c in bytearray(key):
        h = h * 33 + c
    return h & 4294967295