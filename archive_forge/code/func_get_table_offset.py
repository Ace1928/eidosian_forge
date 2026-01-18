import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
def get_table_offset(self, tag_name):
    """ Return the virtual address and file offset of a dynamic table.
        """
    ptr = None
    for tag in self._iter_tags(type=tag_name):
        ptr = tag['d_ptr']
        break
    offset = None
    if ptr:
        offset = next(self.elffile.address_offsets(ptr), None)
    return (ptr, offset)