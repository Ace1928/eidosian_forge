import struct
from ..common.utils import struct_parse
from .sections import Section
@staticmethod
def elf_hash(name):
    """ Compute the hash value for a given symbol name.
        """
    if not isinstance(name, bytes):
        name = name.encode('utf-8')
    h = 0
    x = 0
    for c in bytearray(name):
        h = (h << 4) + c
        x = h & 4026531840
        if x != 0:
            h ^= x >> 24
        h &= ~x
    return h