import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def get_addr(self, cu, addr_index):
    """Provided a CU and an index, retrieves an address from the debug_addr section
        """
    if not self.debug_addr_sec:
        raise DWARFError('The file does not contain a debug_addr section for indirect address access')
    cu_addr_base = _get_base_offset(cu, 'DW_AT_addr_base')
    return struct_parse(cu.structs.Dwarf_target_addr(''), self.debug_addr_sec.stream, cu_addr_base + addr_index * cu.header.address_size)