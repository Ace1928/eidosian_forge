import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def get_range_list_at_offset_ex(self, offset):
    """Get a DWARF v5 range list, addresses and offsets unresolved,
        at the given offset in the section
        """
    return struct_parse(self.structs.Dwarf_rnglists_entries, self.stream, offset)