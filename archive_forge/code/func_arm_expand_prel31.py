from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
def arm_expand_prel31(address, place):
    """
       address: uint32
       place: uint32
       return: uint64
    """
    location = address & 2147483647
    if location & 67108864:
        location |= 18446744071562067968
    return location + place & 18446744073709551615